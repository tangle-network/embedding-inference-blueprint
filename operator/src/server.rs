//! OpenAI-compatible HTTP server for the embedding operator.
//!
//! All shared infrastructure (nonce store, spend-auth validation, x402 headers,
//! metrics, app state container) lives in `tangle-inference-core`. This module
//! only contains:
//!
//! * `EmbeddingBackend` -- the backend attached to `AppState` via `AppStateBuilder`.
//! * Request/response types for the OpenAI embeddings + Cohere rerank wire formats.
//! * HTTP handlers that glue the shared billing flow to the embedding backend.

use blueprint_sdk::std::sync::Arc;
use blueprint_sdk::std::time::Duration;

use axum::{
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router as HttpRouter,
};
use serde::{Deserialize, Serialize};
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use tangle_inference_core::server::{
    acquire_permit, billing_gate, error_response, gpu_health_handler, metrics_handler,
    settle_billing,
};
use tangle_inference_core::{
    detect_gpus, AppState, CostModel, CostParams, PerTokenCostModel, RequestGuard,
};

use crate::config::OperatorConfig;
use crate::embedding::EmbeddingClient;

/// Backend attached to `AppState` via `AppStateBuilder::backend`. Handlers
/// retrieve it via `state.backend::<EmbeddingBackend>().unwrap()`.
pub struct EmbeddingBackend {
    pub config: Arc<OperatorConfig>,
    pub client: Arc<EmbeddingClient>,
    pub cost_model: PerTokenCostModel,
}

impl EmbeddingBackend {
    pub fn new(config: Arc<OperatorConfig>, client: Arc<EmbeddingClient>) -> Self {
        let cost_model = PerTokenCostModel {
            price_per_input_token: config.embedding.price_per_input_token(),
            price_per_output_token: 0,
        };
        Self {
            config,
            client,
            cost_model,
        }
    }

    /// Calculate cost for a request given prompt token count.
    pub fn calculate_cost(&self, prompt_tokens: u32) -> u64 {
        self.cost_model.calculate_cost(&CostParams {
            prompt_tokens,
            completion_tokens: 0,
            ..Default::default()
        })
    }
}

/// Start the HTTP server with graceful shutdown support, returns a join handle.
pub async fn start(
    state: AppState,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<JoinHandle<()>> {
    let backend = state
        .backend::<EmbeddingBackend>()
        .ok_or_else(|| anyhow::anyhow!("AppState backend is not an EmbeddingBackend"))?;
    let max_request_body_bytes = state.server_config.max_request_body_bytes;
    let request_timeout_secs = state.server_config.stream_timeout_secs;
    let bind = format!("{}:{}", state.server_config.host, state.server_config.port);
    let _ = backend;

    let app = HttpRouter::new()
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/rerank", post(rerank))
        .route("/v1/models", get(list_models))
        .route("/v1/operator", get(operator_info))
        .route("/health", get(health_check))
        .route("/health/gpu", get(gpu_health_handler))
        .route("/metrics", get(metrics_handler))
        .layer(DefaultBodyLimit::max(max_request_body_bytes))
        .layer(TimeoutLayer::new(Duration::from_secs(request_timeout_secs)))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&bind).await?;
    tracing::info!(bind = %bind, "HTTP server listening");

    let handle = tokio::spawn(async move {
        let shutdown_signal = async move {
            let _ = shutdown_rx.wait_for(|&v| v).await;
            tracing::info!("HTTP server received shutdown signal");
        };
        if let Err(e) = axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal)
            .await
        {
            tracing::error!(error = %e, "HTTP server error");
        }
    });

    Ok(handle)
}

// --- Request / Response types (OpenAI-compatible) ---

#[derive(Debug, Deserialize)]
pub struct EmbeddingApiRequest {
    /// Text input(s) to embed. Can be a single string or array of strings.
    pub input: EmbeddingInput,

    /// Model identifier (optional, defaults to configured model).
    pub model: Option<String>,

    /// Encoding format: "float" (default) or "base64".
    #[serde(default = "default_encoding_format")]
    pub encoding_format: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

impl EmbeddingInput {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::Multiple(v) => v,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::Multiple(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[derive(Debug, Serialize)]
pub struct EmbeddingApiResponse {
    pub object: String,
    pub data: Vec<EmbeddingDataResponse>,
    pub model: String,
    pub usage: EmbeddingUsageResponse,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingDataResponse {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingUsageResponse {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

// --- Rerank request / response types (Cohere-compatible) ---

#[derive(Debug, Deserialize)]
pub struct RerankApiRequest {
    pub model: Option<String>,
    pub query: String,
    pub documents: Vec<String>,
    pub top_n: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct RerankApiResponse {
    pub results: Vec<RerankResultEntry>,
}

#[derive(Debug, Serialize)]
pub struct RerankResultEntry {
    pub index: usize,
    pub relevance_score: f64,
    pub document: String,
}

#[derive(Debug, Serialize)]
struct ModelInfo {
    id: String,
    object: String,
    owned_by: String,
}

#[derive(Debug, Serialize)]
struct ModelList {
    object: String,
    data: Vec<ModelInfo>,
}

fn default_encoding_format() -> String {
    "float".to_string()
}

// --- Handlers ---

fn backend_from(state: &AppState) -> &EmbeddingBackend {
    state
        .backend::<EmbeddingBackend>()
        .expect("AppState backend is EmbeddingBackend (checked in server::start)")
}

async fn embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<EmbeddingApiRequest>,
) -> Response {
    let backend = backend_from(&state);
    let model_name = req
        .model
        .clone()
        .unwrap_or_else(|| backend.config.embedding.model.clone());
    let mut metrics_guard = RequestGuard::new(&model_name);

    let _permit = match acquire_permit(&state) {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    // Validate batch size
    let input_count = req.input.len();
    let max_batch = backend.config.embedding.max_batch_size;
    if input_count > max_batch {
        return error_response(
            StatusCode::BAD_REQUEST,
            format!("batch size {input_count} exceeds maximum {max_batch}"),
            "invalid_request_error",
            "batch_too_large",
        );
    }
    if input_count == 0 {
        return error_response(
            StatusCode::BAD_REQUEST,
            "input must not be empty".to_string(),
            "invalid_request_error",
            "empty_input",
        );
    }

    // Billing gate: extract x402 -> validate -> authorize on-chain
    let estimated_tokens = (input_count as u32)
        .saturating_mul(backend.config.embedding.max_sequence_length);
    let estimated = backend.calculate_cost(estimated_tokens);
    let (spend_auth, preauth_amount) =
        match billing_gate(&state, &headers, None, estimated).await {
            Ok(v) => v,
            Err(resp) => return resp,
        };

    // Check backend health before serving (only when billing)
    if spend_auth.is_some() && !backend.client.is_healthy().await {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "embedding backend is unavailable -- billing not initiated".to_string(),
            "upstream_error",
            "backend_unhealthy",
        );
    }

    // Forward to embedding backend
    let inputs = req.input.into_vec();
    match backend.client.embed_with_model(inputs, &model_name).await {
        Ok(resp) => {
            let usage = resp.usage.as_ref();
            let prompt_tokens = usage.map(|u| u.prompt_tokens).unwrap_or(0);
            let total_tokens = usage.map(|u| u.total_tokens).unwrap_or(0);

            metrics_guard.set_tokens(prompt_tokens, 0);
            metrics_guard.set_success();

            if let (Some(ref sa), Some(preauth)) = (&spend_auth, preauth_amount) {
                let actual_cost = backend.calculate_cost(prompt_tokens);
                if let Err(e) = settle_billing(&state.billing, sa, preauth, actual_cost).await {
                    tracing::error!(error = %e, "on-chain settlement failed");
                }
            }

            let api_response = EmbeddingApiResponse {
                object: "list".to_string(),
                data: resp
                    .data
                    .into_iter()
                    .map(|d| EmbeddingDataResponse {
                        object: "embedding".to_string(),
                        embedding: d.embedding,
                        index: d.index,
                    })
                    .collect(),
                model: resp.model.unwrap_or(model_name),
                usage: EmbeddingUsageResponse {
                    prompt_tokens,
                    total_tokens,
                },
            };

            (StatusCode::OK, Json(api_response)).into_response()
        }
        Err(e) => {
            tracing::error!(error = %e, "embedding backend request failed");
            error_response(
                StatusCode::BAD_GATEWAY,
                format!("upstream embedding error: {e}"),
                "upstream_error",
                "embedding_error",
            )
        }
    }
}

async fn rerank(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<RerankApiRequest>,
) -> Response {
    let backend = backend_from(&state);

    if !backend
        .config
        .embedding
        .supported_operations
        .iter()
        .any(|op| op == "rerank")
    {
        return error_response(
            StatusCode::NOT_FOUND,
            "reranking is not enabled on this operator".to_string(),
            "invalid_request_error",
            "operation_not_supported",
        );
    }

    let model_name = req
        .model
        .clone()
        .or_else(|| backend.config.embedding.rerank_model.clone())
        .unwrap_or_else(|| backend.config.embedding.model.clone());
    let mut metrics_guard = RequestGuard::new(&model_name);

    let _permit = match acquire_permit(&state) {
        Ok(p) => p,
        Err(resp) => return resp,
    };

    if req.query.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "query must not be empty".to_string(),
            "invalid_request_error",
            "empty_query",
        );
    }
    if req.documents.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "documents must not be empty".to_string(),
            "invalid_request_error",
            "empty_documents",
        );
    }

    // Billing gate
    let estimated_tokens = ((req.documents.len() + 1) as u32)
        .saturating_mul(backend.config.embedding.max_sequence_length);
    let estimated = backend.calculate_cost(estimated_tokens);
    let (spend_auth, preauth_amount) =
        match billing_gate(&state, &headers, None, estimated).await {
            Ok(v) => v,
            Err(resp) => return resp,
        };

    if spend_auth.is_some() && !backend.client.is_healthy().await {
        return error_response(
            StatusCode::SERVICE_UNAVAILABLE,
            "embedding backend is unavailable -- billing not initiated".to_string(),
            "upstream_error",
            "backend_unhealthy",
        );
    }

    // Estimate tokens for cost settlement
    let documents = req.documents.clone();
    let query_len_est = (req.query.len() as u32) / 4 + 1;
    let documents_len_est: u32 = req
        .documents
        .iter()
        .map(|d| (d.len() as u32) / 4 + 1)
        .sum();
    let total_tokens_est = query_len_est.saturating_add(documents_len_est);

    match backend
        .client
        .rerank(req.query, req.documents, &model_name, req.top_n)
        .await
    {
        Ok(resp) => {
            metrics_guard.set_tokens(total_tokens_est, 0);
            metrics_guard.set_success();

            if let (Some(ref sa), Some(preauth)) = (&spend_auth, preauth_amount) {
                let actual_cost = backend.calculate_cost(total_tokens_est);
                if let Err(e) = settle_billing(&state.billing, sa, preauth, actual_cost).await {
                    tracing::error!(error = %e, "on-chain settlement failed");
                }
            }

            let results: Vec<RerankResultEntry> = resp
                .results
                .into_iter()
                .map(|r| RerankResultEntry {
                    index: r.index,
                    relevance_score: r.relevance_score,
                    document: r
                        .document
                        .map(|d| d.text)
                        .unwrap_or_else(|| documents.get(r.index).cloned().unwrap_or_default()),
                })
                .collect();

            (StatusCode::OK, Json(RerankApiResponse { results })).into_response()
        }
        Err(e) => {
            tracing::error!(error = %e, "rerank backend request failed");
            error_response(
                StatusCode::BAD_GATEWAY,
                format!("upstream rerank error: {e}"),
                "upstream_error",
                "rerank_error",
            )
        }
    }
}

async fn list_models(State(state): State<AppState>) -> Json<ModelList> {
    let backend = backend_from(&state);
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: backend.config.embedding.model.clone(),
            object: "model".to_string(),
            owned_by: "operator".to_string(),
        }],
    })
}

/// Operator info endpoint for discovery.
async fn operator_info(State(state): State<AppState>) -> Json<serde_json::Value> {
    let backend = backend_from(&state);
    let gpu_info = detect_gpus().await.unwrap_or_default();
    Json(serde_json::json!({
        "operator": format!("{:#x}", state.operator_address),
        "model": backend.config.embedding.model,
        "dimensions": backend.config.embedding.dimensions,
        "max_sequence_length": backend.config.embedding.max_sequence_length,
        "supported_operations": backend.config.embedding.supported_operations,
        "pricing": {
            "price_per_1k_tokens": backend.config.embedding.price_per_1k_tokens,
            "currency": "tsUSD",
        },
        "gpu": {
            "count": backend.config.gpu.expected_gpu_count,
            "min_vram_mib": backend.config.gpu.min_vram_mib,
            "model": backend.config.gpu.gpu_model,
            "detected": gpu_info,
        },
        "server": {
            "max_concurrent_requests": state.server_config.max_concurrent_requests,
            "max_batch_size": backend.config.embedding.max_batch_size,
        },
        "billing_required": state.billing_config.billing_required,
        "payment_token": state.billing_config.payment_token_address,
    }))
}

async fn health_check(State(state): State<AppState>) -> Response {
    let backend = backend_from(&state);
    if backend.client.is_healthy().await {
        let body = serde_json::json!({
            "status": "ok",
            "model": backend.config.embedding.model,
            "dimensions": backend.config.embedding.dimensions,
        });
        (StatusCode::OK, Json(body)).into_response()
    } else {
        let body = serde_json::json!({
            "status": "degraded",
            "model": backend.config.embedding.model,
            "backend": "unhealthy",
        });
        (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
    }
}
