//! Embedding-specific operator configuration.
//!
//! Shared infrastructure config (`TangleConfig`, `ServerConfig`, `BillingConfig`,
//! `GpuConfig`) lives in `tangle-inference-core` and is re-exported here for
//! convenience.

use serde::{Deserialize, Serialize};

pub use tangle_inference_core::{BillingConfig, GpuConfig, ServerConfig, TangleConfig};

use tangle_inference_core::qos::QoSConfig;

/// Top-level operator configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    /// Tangle network configuration (shared).
    pub tangle: TangleConfig,

    /// Embedding backend + per-token pricing configuration (embedding-specific).
    pub embedding: EmbeddingConfig,

    /// HTTP server configuration (shared).
    pub server: ServerConfig,

    /// Billing / ShieldedCredits infrastructure configuration (shared).
    pub billing: BillingConfig,

    /// GPU configuration (shared).
    pub gpu: GpuConfig,

    /// QoS heartbeat configuration (optional -- disabled by default).
    #[serde(default)]
    pub qos: Option<QoSConfig>,
}

/// Embedding backend + pricing config. This is the only truly embedding-specific
/// config section -- everything else comes from `tangle-inference-core`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model identifier (e.g. "BAAI/bge-large-en-v1.5")
    pub model: String,

    /// Embedding vector dimension (e.g. 1024 for bge-large)
    pub dimensions: u32,

    /// Maximum sequence length in tokens
    pub max_sequence_length: u32,

    /// Maximum batch size (number of inputs per request)
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    /// Endpoint of the embedding backend (TEI or OpenAI-compatible).
    /// Overridden by EMBEDDING_ENDPOINT env var.
    #[serde(default = "default_embedding_endpoint")]
    pub endpoint: String,

    /// Separate endpoint for reranking (falls back to embedding endpoint).
    /// Overridden by RERANK_ENDPOINT env var.
    #[serde(default)]
    pub rerank_endpoint: Option<String>,

    /// Model for reranking (e.g. "BAAI/bge-reranker-v2-m3").
    /// If unset, rerank requests must specify a model explicitly.
    #[serde(default)]
    pub rerank_model: Option<String>,

    /// Supported operations: ["embed"], ["embed", "rerank"], etc.
    #[serde(default = "default_supported_operations")]
    pub supported_operations: Vec<String>,

    /// HuggingFace token for gated models.
    pub hf_token: Option<String>,

    /// Startup timeout in seconds (for health-check polling).
    #[serde(default = "default_startup_timeout")]
    pub startup_timeout_secs: u64,

    /// Price per 1K input tokens in base token units.
    pub price_per_1k_tokens: u64,
}

impl EmbeddingConfig {
    /// Convert per-1K-token price into the per-token price used by
    /// `PerTokenCostModel`. Embeddings only consume input tokens, so the
    /// output-token price is zero.
    pub fn price_per_input_token(&self) -> u64 {
        // Round up to avoid ever charging zero on non-empty requests.
        self.price_per_1k_tokens.div_ceil(1000)
    }
}

fn default_embedding_endpoint() -> String {
    "http://127.0.0.1:8080".to_string()
}

fn default_supported_operations() -> Vec<String> {
    vec!["embed".to_string()]
}

fn default_max_batch_size() -> usize {
    64
}

fn default_startup_timeout() -> u64 {
    120
}

impl OperatorConfig {
    /// Load config from file, env vars, and CLI overrides.
    pub fn load(path: Option<&str>) -> anyhow::Result<Self> {
        let mut builder = config::Config::builder();

        if let Some(path) = path {
            builder = builder.add_source(config::File::with_name(path));
        }

        // Environment variables override file config.
        // Prefix: EMBED_OP_ (e.g. EMBED_OP_TANGLE__RPC_URL)
        builder = builder.add_source(
            config::Environment::with_prefix("EMBED_OP")
                .prefix_separator("_")
                .separator("__")
                .try_parsing(true),
        );

        // EMBEDDING_ENDPOINT env var overrides embedding.endpoint
        if let Ok(endpoint) = std::env::var("EMBEDDING_ENDPOINT") {
            builder = builder.set_override("embedding.endpoint", endpoint)?;
        }

        // RERANK_ENDPOINT env var overrides embedding.rerank_endpoint
        if let Ok(endpoint) = std::env::var("RERANK_ENDPOINT") {
            builder = builder.set_override("embedding.rerank_endpoint", endpoint)?;
        }

        let cfg = builder.build()?.try_deserialize::<Self>()?;
        Ok(cfg)
    }
}
