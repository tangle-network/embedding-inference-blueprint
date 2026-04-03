use blueprint_sdk::std::collections::HashMap;
use blueprint_sdk::std::path::PathBuf;
use blueprint_sdk::std::sync::{Arc, RwLock};
use blueprint_sdk::std::time::Duration;

use alloy::primitives::Address;
use axum::{
    body::Body,
    extract::{DefaultBodyLimit, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router as HttpRouter,
};
use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;
use tower_http::cors::CorsLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;

use crate::config::OperatorConfig;
use crate::embedding::EmbeddingBackend;
use crate::health;

/// Nonce key: (commitment, nonce) pair. Prevents replay of SpendAuth signatures.
type NonceKey = (String, u64);

// --- x402 constants ---

const X402_PAYMENT_REQUIRED: &str = "X-Payment-Required";
const X402_PAYMENT_TOKEN: &str = "X-Payment-Token";
const X402_PAYMENT_RECIPIENT: &str = "X-Payment-Recipient";
const X402_PAYMENT_NETWORK: &str = "X-Payment-Network";
const X402_PAYMENT_SIGNATURE: &str = "X-Payment-Signature";

// --- Persistent Nonce Store ---

#[derive(Serialize, Deserialize)]
struct NonceRecord {
    commitment: String,
    nonce: u64,
    expiry: u64,
}

/// Replay-protection nonce store with optional file persistence.
pub struct NonceStore {
    nonces: RwLock<HashMap<NonceKey, u64>>,
    path: Option<PathBuf>,
}

impl NonceStore {
    pub fn load(path: Option<PathBuf>) -> Self {
        let nonces: HashMap<NonceKey, u64> = path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|data| serde_json::from_str::<Vec<NonceRecord>>(&data).ok())
            .map(|records| {
                records
                    .into_iter()
                    .map(|r| ((r.commitment, r.nonce), r.expiry))
                    .collect()
            })
            .unwrap_or_default();

        if path.is_some() {
            tracing::info!(count = nonces.len(), "loaded persisted nonces");
        } else {
            tracing::warn!(
                "nonce_store_path not configured -- nonces are in-memory only"
            );
        }

        Self {
            nonces: RwLock::new(nonces),
            path,
        }
    }

    pub fn check_replay(&self, key: &NonceKey, tolerance: u64) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let mut nonces = self.nonces.write().unwrap_or_else(|e| e.into_inner());
        nonces.retain(|_, expiry| now <= expiry.saturating_add(tolerance));
        nonces.contains_key(key)
    }

    pub fn insert(&self, key: NonceKey, expiry: u64, tolerance: u64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let mut nonces = self.nonces.write().unwrap_or_else(|e| e.into_inner());
        nonces.retain(|_, exp| now <= exp.saturating_add(tolerance));
        nonces.insert(key, expiry);
        self.persist(&nonces);
    }

    fn persist(&self, nonces: &HashMap<NonceKey, u64>) {
        let Some(ref path) = self.path else { return };
        let records: Vec<NonceRecord> = nonces
            .iter()
            .map(|((commitment, nonce), expiry)| NonceRecord {
                commitment: commitment.clone(),
                nonce: *nonce,
                expiry: *expiry,
            })
            .collect();
        let Ok(data) = serde_json::to_string(&records) else {
            tracing::error!("failed to serialize nonce store");
            return;
        };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let tmp = path.with_extension("tmp");
        if std::fs::write(&tmp, &data).is_ok() {
            if let Err(e) = std::fs::rename(&tmp, path) {
                tracing::warn!(error = %e, "failed to persist nonce store");
            }
        }
    }
}

// --- Per-Account Concurrency Guard ---

struct AccountGuard {
    commitment: String,
    active: Arc<RwLock<HashMap<String, usize>>>,
}

impl Drop for AccountGuard {
    fn drop(&mut self) {
        let mut map = self.active.write().unwrap_or_else(|e| e.into_inner());
        if let Some(count) = map.get_mut(&self.commitment) {
            *count = count.saturating_sub(1);
            if *count == 0 {
                map.remove(&self.commitment);
            }
        }
    }
}

/// Shared application state for the HTTP server.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<OperatorConfig>,
    pub backend: Arc<EmbeddingBackend>,
    pub semaphore: Arc<Semaphore>,
    pub nonce_store: Arc<NonceStore>,
    pub active_per_account: Arc<RwLock<HashMap<String, usize>>>,
    pub operator_address: Address,
}

/// Start the HTTP server with graceful shutdown support.
pub async fn start(
    state: AppState,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> anyhow::Result<JoinHandle<()>> {
    let app = HttpRouter::new()
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/rerank", post(rerank))
        .route("/v1/models", get(list_models))
        .route("/health", get(health_check))
        .route("/health/gpu", get(gpu_health))
        .layer(DefaultBodyLimit::max(
            state.config.server.max_request_body_bytes,
        ))
        .layer(TimeoutLayer::new(Duration::from_secs(
            state.config.server.request_timeout_secs,
        )))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state.clone());

    let bind = format!("{}:{}", state.config.server.host, state.config.server.port);
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

    /// SpendAuth for billing.
    pub spend_auth: Option<SpendAuthPayload>,
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
}

#[derive(Debug, Deserialize)]
pub struct SpendAuthPayload {
    pub commitment: String,
    pub service_id: u64,
    pub job_index: u8,
    pub amount: String,
    pub operator: String,
    pub nonce: u64,
    pub expiry: u64,
    pub signature: String,
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
    /// Model identifier (optional, falls back to configured rerank_model).
    pub model: Option<String>,

    /// Query to rank documents against.
    pub query: String,

    /// Documents to rerank.
    pub documents: Vec<String>,

    /// Return only the top N results.
    pub top_n: Option<usize>,

    /// SpendAuth for billing.
    pub spend_auth: Option<SpendAuthPayload>,
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

#[derive(Debug, Serialize)]
pub(crate) struct ErrorResponse {
    pub error: ErrorDetail,
}

#[derive(Debug, Serialize)]
pub(crate) struct ErrorDetail {
    pub message: String,
    pub r#type: String,
    pub code: String,
}

fn default_encoding_format() -> String {
    "float".to_string()
}

fn error_response(status: StatusCode, message: String, error_type: &str, code: &str) -> Response {
    let body = ErrorResponse {
        error: ErrorDetail {
            message,
            r#type: error_type.to_string(),
            code: code.to_string(),
        },
    };
    (status, Json(body)).into_response()
}

// --- x402 Payment Required response ---

fn x402_payment_required(state: &AppState) -> Response {
    let estimated_amount = state.config.billing.min_charge_amount.max(
        state.config.calculate_cost(1000), // estimate 1K tokens
    );

    let body = serde_json::json!({
        "error": "payment_required",
        "amount": estimated_amount.to_string(),
        "token": state.config.billing.payment_token_address.as_deref().unwrap_or("0x0000000000000000000000000000000000000000"),
        "recipient": format!("{}", state.operator_address),
        "network": state.config.tangle.chain_id.to_string(),
        "accepts": ["spend_auth"],
        "description": "ShieldedCredits SpendAuth required. Include spend_auth in request body or X-Payment-Signature header."
    });

    Response::builder()
        .status(StatusCode::PAYMENT_REQUIRED)
        .header(header::CONTENT_TYPE, "application/json")
        .header(X402_PAYMENT_REQUIRED, estimated_amount.to_string())
        .header(
            X402_PAYMENT_TOKEN,
            state
                .config
                .billing
                .payment_token_address
                .as_deref()
                .unwrap_or("0x0000000000000000000000000000000000000000"),
        )
        .header(
            X402_PAYMENT_RECIPIENT,
            format!("{}", state.operator_address),
        )
        .header(
            X402_PAYMENT_NETWORK,
            state.config.tangle.chain_id.to_string(),
        )
        .body(Body::from(serde_json::to_string(&body).unwrap_or_default()))
        .unwrap_or_else(|e| {
            error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("failed to build 402 response: {e}"),
                "internal_error",
                "response_build_failed",
            )
        })
}

/// Try to extract a SpendAuth from x402 headers.
fn extract_x402_spend_auth(headers: &HeaderMap) -> Option<SpendAuthPayload> {
    let header_val = headers.get(X402_PAYMENT_SIGNATURE)?.to_str().ok()?;

    if let Ok(payload) = serde_json::from_str::<SpendAuthPayload>(header_val) {
        return Some(payload);
    }

    // Try hex-encoded JSON
    use alloy::primitives::hex;
    if let Ok(decoded) = hex::decode(header_val.strip_prefix("0x").unwrap_or(header_val)) {
        if let Ok(payload) = serde_json::from_slice::<SpendAuthPayload>(&decoded) {
            return Some(payload);
        }
    }

    None
}

// --- Handlers ---

async fn embeddings(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(mut req): Json<EmbeddingApiRequest>,
) -> Response {
    // 1. Acquire semaphore permit
    let _permit: OwnedSemaphorePermit = match state.semaphore.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return error_response(
                StatusCode::TOO_MANY_REQUESTS,
                "server at capacity".to_string(),
                "rate_limit_error",
                "too_many_requests",
            );
        }
    };

    // 2. Validate batch size
    let input_count = req.input.len();
    let max_batch = state.config.embedding.max_batch_size;
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

    // 3. x402 flow: check for spend_auth in headers if not in body
    if req.spend_auth.is_none() {
        if let Some(x402_auth) = extract_x402_spend_auth(&headers) {
            req.spend_auth = Some(x402_auth);
        }
    }

    // 4. Enforce billing requirement
    if state.config.billing.billing_required && req.spend_auth.is_none() {
        return x402_payment_required(&state);
    }

    // 5. Validate SpendAuth if present
    if let Some(ref spend_auth) = req.spend_auth {
        // 5a. Parse amount
        let requested_amount: u64 = match spend_auth.amount.parse() {
            Ok(v) => v,
            Err(_) => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid spend_auth amount: must be a valid u64 integer".to_string(),
                    "billing_error",
                    "invalid_amount",
                );
            }
        };

        // 5b. Enforce min_charge_amount
        let min_charge = state.config.billing.min_charge_amount;
        if min_charge > 0 && requested_amount < min_charge {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "spend authorization amount ({requested_amount}) is below minimum charge ({min_charge})"
                ),
                "billing_error",
                "below_min_charge",
            );
        }

        // 5c. Enforce max_spend_per_request
        let max_spend = state.config.billing.max_spend_per_request;
        if max_spend > 0 && requested_amount > max_spend {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "spend authorization amount ({requested_amount}) exceeds max_spend_per_request ({max_spend})"
                ),
                "billing_error",
                "exceeds_max_spend",
            );
        }

        // 5d. Validate operator matches
        let spend_operator: Address = match spend_auth.operator.parse() {
            Ok(addr) => addr,
            Err(_) => {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    "invalid operator address in spend_auth".to_string(),
                    "billing_error",
                    "invalid_operator",
                );
            }
        };
        if spend_operator != state.operator_address {
            return error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "spend_auth operator ({spend_operator}) does not match this operator ({})",
                    state.operator_address
                ),
                "billing_error",
                "operator_mismatch",
            );
        }

        // 5e. Validate service_id
        if let Some(expected_service_id) = state.config.tangle.service_id {
            if spend_auth.service_id != expected_service_id {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "spend_auth service_id ({}) does not match operator service ({expected_service_id})",
                        spend_auth.service_id
                    ),
                    "billing_error",
                    "service_id_mismatch",
                );
            }
        }

        // 5f. Nonce replay protection
        let nonce_key = (spend_auth.commitment.clone(), spend_auth.nonce);
        if state
            .nonce_store
            .check_replay(&nonce_key, state.config.billing.clock_skew_tolerance_secs)
        {
            return error_response(
                StatusCode::BAD_REQUEST,
                "spend_auth nonce already used (replay detected)".to_string(),
                "billing_error",
                "nonce_replay",
            );
        }

        // 5g. Per-account concurrency limit
        let max_per_account = state.config.server.max_per_account_requests;
        if max_per_account > 0 {
            let mut map = state
                .active_per_account
                .write()
                .unwrap_or_else(|e| e.into_inner());
            let count = map.entry(spend_auth.commitment.clone()).or_insert(0);
            if *count >= max_per_account {
                return error_response(
                    StatusCode::TOO_MANY_REQUESTS,
                    format!("account has {count} active requests (limit: {max_per_account})"),
                    "rate_limit_error",
                    "per_account_limit",
                );
            }
            *count += 1;
            // AccountGuard will decrement on drop
        }

        // 5h. Check backend health before committing
        if !state.backend.is_healthy().await {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "embedding backend is unavailable".to_string(),
                "upstream_error",
                "backend_unhealthy",
            );
        }

        // Record nonce as used
        let nonce_key = (spend_auth.commitment.clone(), spend_auth.nonce);
        state.nonce_store.insert(
            nonce_key,
            spend_auth.expiry,
            state.config.billing.clock_skew_tolerance_secs,
        );
    }

    // 6. Forward to embedding backend
    let inputs = req.input.into_vec();
    let model = req
        .model
        .as_deref()
        .unwrap_or(state.backend.model());

    let result = state.backend.embed_with_model(inputs, model).await;

    match result {
        Ok(resp) => {
            let usage = resp.usage.as_ref();
            let prompt_tokens = usage.map(|u| u.prompt_tokens).unwrap_or(0);
            let total_tokens = usage.map(|u| u.total_tokens).unwrap_or(0);

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
                model: resp.model.unwrap_or_else(|| state.backend.model().to_string()),
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
    Json(mut req): Json<RerankApiRequest>,
) -> Response {
    // Check that rerank is a supported operation
    if !state
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

    // Acquire semaphore permit
    let _permit: OwnedSemaphorePermit = match state.semaphore.clone().try_acquire_owned() {
        Ok(p) => p,
        Err(_) => {
            return error_response(
                StatusCode::TOO_MANY_REQUESTS,
                "server at capacity".to_string(),
                "rate_limit_error",
                "too_many_requests",
            );
        }
    };

    // Validate input
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

    // x402 flow
    if req.spend_auth.is_none() {
        if let Some(x402_auth) = extract_x402_spend_auth(&headers) {
            req.spend_auth = Some(x402_auth);
        }
    }

    if state.config.billing.billing_required && req.spend_auth.is_none() {
        return x402_payment_required(&state);
    }

    // Resolve model
    let model = req
        .model
        .as_deref()
        .or(state.config.embedding.rerank_model.as_deref())
        .unwrap_or(state.backend.model());

    // Forward to reranking backend
    let documents = req.documents.clone();
    let result = state
        .backend
        .rerank(req.query, req.documents, model, req.top_n)
        .await;

    match result {
        Ok(resp) => {
            let results: Vec<RerankResultEntry> = resp
                .results
                .into_iter()
                .map(|r| RerankResultEntry {
                    index: r.index,
                    relevance_score: r.relevance_score,
                    document: r
                        .document
                        .map(|d| d.text)
                        .unwrap_or_else(|| {
                            documents.get(r.index).cloned().unwrap_or_default()
                        }),
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
    Json(ModelList {
        object: "list".to_string(),
        data: vec![ModelInfo {
            id: state.config.embedding.model.clone(),
            object: "model".to_string(),
            owned_by: "operator".to_string(),
        }],
    })
}

async fn health_check(State(state): State<AppState>) -> Response {
    let backend_healthy = state.backend.is_healthy().await;

    if backend_healthy {
        let body = serde_json::json!({
            "status": "ok",
            "model": state.config.embedding.model,
            "dimensions": state.config.embedding.dimensions,
        });
        (StatusCode::OK, Json(body)).into_response()
    } else {
        let body = serde_json::json!({
            "status": "degraded",
            "model": state.config.embedding.model,
            "backend": "unhealthy",
        });
        (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
    }
}

async fn gpu_health() -> Response {
    match health::detect_gpus().await {
        Ok(gpus) => {
            let body = serde_json::json!({
                "status": "ok",
                "gpus": gpus,
            });
            (StatusCode::OK, Json(body)).into_response()
        }
        Err(e) => {
            let body = serde_json::json!({
                "status": "error",
                "error": e.to_string(),
            });
            (StatusCode::SERVICE_UNAVAILABLE, Json(body)).into_response()
        }
    }
}
