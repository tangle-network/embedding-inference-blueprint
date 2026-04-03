pub mod config;
pub mod embedding;
pub mod health;
pub mod qos;
pub mod server;

use blueprint_sdk::std::sync::{Arc, OnceLock};
use blueprint_sdk::std::time::Duration;

use alloy_sol_types::sol;
use blueprint_sdk::macros::debug_job;
use blueprint_sdk::router::Router;
use blueprint_sdk::runner::error::RunnerError;
use blueprint_sdk::runner::BackgroundService;
use blueprint_sdk::tangle::extract::{TangleArg, TangleResult};
use blueprint_sdk::tangle::layers::TangleLayer;
use blueprint_sdk::Job;
use tokio::sync::oneshot;

use crate::config::OperatorConfig;
use crate::embedding::EmbeddingBackend;

// --- ABI types for on-chain job encoding ---

sol! {
    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    /// Input payload ABI-encoded in the Tangle job call.
    struct EmbeddingRequest {
        /// Array of text inputs to embed (ABI-encoded as a single concatenated
        /// string with newline separators for simplicity; real usage may
        /// use bytes[] or string[]).
        string[] inputs;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    /// Output payload ABI-encoded in the Tangle job result.
    struct EmbeddingResult {
        /// Number of embeddings produced
        uint32 count;
        /// Total tokens consumed
        uint32 totalTokens;
        /// Embedding dimensions
        uint32 dimensions;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    /// Input payload for an on-chain rerank job.
    struct RerankRequest {
        /// Query to rank documents against
        string query;
        /// Documents to rerank
        string[] documents;
        /// Return only top N results (0 = all)
        uint32 topN;
    }

    #[derive(Debug, serde::Serialize, serde::Deserialize)]
    /// Output payload for an on-chain rerank job.
    struct RerankResult {
        /// Number of results returned
        uint32 count;
    }
}

// --- Job IDs ---

pub const EMBEDDING_JOB: u8 = 0;
pub const RERANK_JOB: u8 = 1;

// --- Shared state for the on-chain job handler ---

static EMBEDDING_ENDPOINT: OnceLock<EmbeddingEndpoint> = OnceLock::new();

struct EmbeddingEndpoint {
    backend: Arc<EmbeddingBackend>,
}

fn register_embedding_endpoint(backend: Arc<EmbeddingBackend>) -> Result<(), RunnerError> {
    let endpoint = EmbeddingEndpoint { backend };
    let _ = EMBEDDING_ENDPOINT.set(endpoint);
    Ok(())
}

/// Initialize the embedding endpoint for testing.
pub fn init_for_testing(base_url: &str, model: &str) {
    let backend = EmbeddingBackend::connect(base_url.to_string(), model.to_string())
        .expect("failed to create test backend");
    let endpoint = EmbeddingEndpoint {
        backend: Arc::new(backend),
    };
    let _ = EMBEDDING_ENDPOINT.set(endpoint);
}

// --- Router ---

pub fn router() -> Router {
    Router::new()
        .route(
            EMBEDDING_JOB,
            run_embedding
                .layer(TangleLayer)
                .layer(blueprint_sdk::tee::TeeLayer::new()),
        )
        .route(
            RERANK_JOB,
            run_rerank
                .layer(TangleLayer)
                .layer(blueprint_sdk::tee::TeeLayer::new()),
        )
}

/// Direct embedding call -- same logic as run_embedding but without TangleArg.
/// Used for testing without the Tangle context.
pub async fn embed_direct(request: &EmbeddingRequest) -> Result<EmbeddingResult, RunnerError> {
    let endpoint = EMBEDDING_ENDPOINT.get().ok_or_else(|| {
        RunnerError::Other("embedding endpoint not registered".into())
    })?;

    let inputs: Vec<String> = request.inputs.iter().cloned().collect();

    if inputs.is_empty() {
        return Err(RunnerError::Other("empty input list".into()));
    }

    let resp = endpoint
        .backend
        .embed(inputs)
        .await
        .map_err(|e| RunnerError::Other(format!("embedding request failed: {e}").into()))?;

    let count = resp.data.len() as u32;
    let total_tokens = resp.usage.as_ref().map(|u| u.total_tokens).unwrap_or(0);
    let dimensions = resp.data.first().map(|d| d.embedding.len() as u32).unwrap_or(0);

    Ok(EmbeddingResult {
        count,
        totalTokens: total_tokens,
        dimensions,
    })
}

// --- Job handler ---

/// Handle an embedding job submitted on-chain.
#[debug_job]
pub async fn run_embedding(
    TangleArg(request): TangleArg<EmbeddingRequest>,
) -> Result<TangleResult<EmbeddingResult>, RunnerError> {
    let endpoint = EMBEDDING_ENDPOINT.get().ok_or_else(|| {
        RunnerError::Other(
            "embedding endpoint not registered -- EmbeddingServer not started".into(),
        )
    })?;

    let inputs: Vec<String> = request.inputs.into_iter().collect();

    if inputs.is_empty() {
        return Err(RunnerError::Other("empty input list".into()));
    }

    let resp = endpoint
        .backend
        .embed(inputs)
        .await
        .map_err(|e| {
            tracing::error!(error = %e, "embedding request failed");
            RunnerError::Other(format!("embedding request failed: {e}").into())
        })?;

    let count = resp.data.len() as u32;
    let total_tokens = resp
        .usage
        .as_ref()
        .map(|u| u.total_tokens)
        .unwrap_or(0);
    let dimensions = resp
        .data
        .first()
        .map(|d| d.embedding.len() as u32)
        .unwrap_or(0);

    Ok(TangleResult(EmbeddingResult {
        count,
        totalTokens: total_tokens,
        dimensions,
    }))
}

/// Handle a rerank job submitted on-chain.
#[debug_job]
pub async fn run_rerank(
    TangleArg(request): TangleArg<RerankRequest>,
) -> Result<TangleResult<RerankResult>, RunnerError> {
    let endpoint = EMBEDDING_ENDPOINT.get().ok_or_else(|| {
        RunnerError::Other(
            "embedding endpoint not registered -- EmbeddingServer not started".into(),
        )
    })?;

    let documents: Vec<String> = request.documents.into_iter().collect();
    let query = request.query;
    let top_n = if request.topN == 0 {
        None
    } else {
        Some(request.topN as usize)
    };

    if documents.is_empty() {
        return Err(RunnerError::Other("empty documents list".into()));
    }

    // Use the rerank model from config or fall back to the embedding model
    let model = endpoint.backend.model();

    let resp = endpoint
        .backend
        .rerank(query, documents, model, top_n)
        .await
        .map_err(|e| {
            tracing::error!(error = %e, "rerank request failed");
            RunnerError::Other(format!("rerank request failed: {e}").into())
        })?;

    let count = resp.results.len() as u32;

    Ok(TangleResult(RerankResult { count }))
}

// --- Background service: embedding backend + HTTP server ---

/// Runs the embedding backend health check and the OpenAI-compatible HTTP API
/// as a [`BackgroundService`]. Starts before the BlueprintRunner begins polling
/// for on-chain jobs.
#[derive(Clone)]
pub struct EmbeddingServer {
    pub config: Arc<OperatorConfig>,
}

impl BackgroundService for EmbeddingServer {
    async fn start(&self) -> Result<oneshot::Receiver<Result<(), RunnerError>>, RunnerError> {
        let (tx, rx) = oneshot::channel();
        let config = self.config.clone();

        tokio::spawn(async move {
            // 1. Create embedding backend client
            let backend = match embedding::create_backend(config.clone()) {
                Ok(b) => b,
                Err(e) => {
                    tracing::error!(error = %e, "failed to create embedding backend");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            };

            // 2. Wait for backend readiness
            tracing::info!(
                endpoint = %config.embedding.endpoint,
                "waiting for embedding backend readiness"
            );
            if let Err(e) = backend.wait_ready(&config).await {
                tracing::error!(error = %e, "embedding backend failed to become ready");
                let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                return;
            }
            tracing::info!("embedding backend is ready");

            // 3. Register for on-chain job handlers
            if let Err(e) = register_embedding_endpoint(backend.clone()) {
                tracing::error!(error = %e, "failed to register embedding endpoint");
                let _ = tx.send(Err(e));
                return;
            }

            // 4. Build semaphore
            let max_concurrent = config.server.max_concurrent_requests;
            let semaphore = Arc::new(if max_concurrent == 0 {
                tokio::sync::Semaphore::new(tokio::sync::Semaphore::MAX_PERMITS)
            } else {
                tokio::sync::Semaphore::new(max_concurrent)
            });

            // 5. Create shutdown channel
            let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

            // 6. Build app state and start HTTP server
            // Derive operator address from key
            use alloy::signers::local::PrivateKeySigner;
            use alloy::signers::Signer;
            let signer: PrivateKeySigner = match config.tangle.operator_key.parse() {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(error = %e, "failed to parse operator key");
                    let _ = tx.send(Err(RunnerError::Other(
                        format!("invalid operator key: {e}").into(),
                    )));
                    return;
                }
            };
            let operator_address = signer.address();

            let nonce_store = Arc::new(server::NonceStore::load(
                config.billing.nonce_store_path.clone(),
            ));
            let state = server::AppState {
                config: config.clone(),
                backend,
                semaphore,
                nonce_store,
                active_per_account: Arc::new(std::sync::RwLock::new(
                    std::collections::HashMap::new(),
                )),
                operator_address,
            };

            match server::start(state, shutdown_rx).await {
                Ok(_join_handle) => {
                    tracing::info!("HTTP server started -- background service ready");
                    let _ = tx.send(Ok(()));
                }
                Err(e) => {
                    tracing::error!(error = %e, "failed to start HTTP server");
                    let _ = tx.send(Err(RunnerError::Other(e.to_string().into())));
                    return;
                }
            }

            // 7. Watchdog loop: monitor embedding backend health
            loop {
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(30)) => {}
                    _ = tokio::signal::ctrl_c() => {
                        tracing::info!("received shutdown signal");
                        let _ = shutdown_tx.send(true);
                        return;
                    }
                }

                if !EMBEDDING_ENDPOINT
                    .get()
                    .map(|e| futures::executor::block_on(e.backend.is_healthy()))
                    .unwrap_or(false)
                {
                    tracing::warn!("embedding backend health check failed");
                }
            }
        });

        Ok(rx)
    }
}
