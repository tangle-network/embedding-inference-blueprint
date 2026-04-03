use serde::{Deserialize, Serialize};
use blueprint_std::fmt;
use blueprint_std::path::PathBuf;

use crate::qos::QoSConfig;

/// Top-level operator configuration.
#[derive(Clone, Serialize, Deserialize)]
pub struct OperatorConfig {
    /// Tangle network configuration
    pub tangle: TangleConfig,

    /// Embedding backend configuration
    pub embedding: EmbeddingConfig,

    /// HTTP server configuration
    pub server: ServerConfig,

    /// Billing / ShieldedCredits configuration
    pub billing: BillingConfig,

    /// GPU configuration
    pub gpu: GpuConfig,

    /// QoS heartbeat configuration (optional -- disabled by default)
    #[serde(default)]
    pub qos: Option<QoSConfig>,
}

impl fmt::Debug for OperatorConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OperatorConfig")
            .field("tangle", &self.tangle)
            .field("embedding", &self.embedding)
            .field("server", &self.server)
            .field("billing", &self.billing)
            .field("gpu", &self.gpu)
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TangleConfig {
    /// JSON-RPC endpoint for the Tangle EVM chain
    pub rpc_url: String,

    /// Chain ID
    pub chain_id: u64,

    /// Operator's private key (hex, without 0x prefix)
    pub operator_key: String,

    /// Tangle core contract address
    pub tangle_core: String,

    /// ShieldedCredits contract address
    pub shielded_credits: String,

    /// Blueprint ID this operator is registered for
    pub blueprint_id: u64,

    /// Service ID (set after service activation)
    pub service_id: Option<u64>,
}

impl fmt::Debug for TangleConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TangleConfig")
            .field("rpc_url", &self.rpc_url)
            .field("chain_id", &self.chain_id)
            .field("operator_key", &"[REDACTED]")
            .field("tangle_core", &self.tangle_core)
            .field("shielded_credits", &self.shielded_credits)
            .field("blueprint_id", &self.blueprint_id)
            .field("service_id", &self.service_id)
            .finish()
    }
}

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

    /// Endpoint of the embedding backend (TEI or OpenAI-compatible)
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

    /// HuggingFace token for gated models
    pub hf_token: Option<String>,

    /// Startup timeout in seconds (for health check polling)
    #[serde(default = "default_startup_timeout")]
    pub startup_timeout_secs: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// External host to bind
    #[serde(default = "default_host")]
    pub host: String,

    /// External port to bind
    #[serde(default = "default_port")]
    pub port: u16,

    /// Maximum concurrent requests
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_requests: usize,

    /// Maximum request body size in bytes (default 2 MiB)
    #[serde(default = "default_max_request_body_bytes")]
    pub max_request_body_bytes: usize,

    /// Per-request timeout in seconds (default 60)
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,

    /// Maximum concurrent requests per credit account (commitment).
    /// 0 = unlimited (default).
    #[serde(default)]
    pub max_per_account_requests: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BillingConfig {
    /// Whether billing is required for HTTP requests.
    #[serde(default = "default_billing_required")]
    pub required: bool,

    /// Price per 1K input tokens in tsUSD base units (6 decimals)
    pub price_per_1k_tokens: u64,

    /// Maximum amount a single SpendAuth can authorize (anti-abuse)
    pub max_spend_per_request: u64,

    /// Minimum balance required in a credit account to serve a request
    pub min_credit_balance: u64,

    /// Whether billing (spend_auth) is required on every request.
    #[serde(default = "default_billing_required")]
    pub billing_required: bool,

    /// Minimum charge amount per request (gas cost protection).
    #[serde(default)]
    pub min_charge_amount: u64,

    /// Maximum retries for claim_payment on-chain calls.
    #[serde(default = "default_claim_max_retries")]
    pub claim_max_retries: u32,

    /// Clock skew tolerance in seconds for SpendAuth expiry checks.
    #[serde(default = "default_clock_skew_tolerance")]
    pub clock_skew_tolerance_secs: u64,

    /// Maximum gas price in gwei for billing txs. 0 = no cap.
    #[serde(default)]
    pub max_gas_price_gwei: u64,

    /// Path to persist used nonces across restarts.
    #[serde(default = "default_nonce_store_path")]
    pub nonce_store_path: Option<PathBuf>,

    /// ERC-20 token address for x402 payment (e.g. tsUSD).
    #[serde(default)]
    pub payment_token_address: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Expected number of GPUs
    pub expected_gpu_count: u32,

    /// Minimum required VRAM per GPU in MiB
    pub min_vram_mib: u32,

    /// GPU model name for on-chain registration
    #[serde(default)]
    pub gpu_model: Option<String>,

    /// GPU monitoring interval in seconds
    #[serde(default = "default_monitor_interval")]
    pub monitor_interval_secs: u64,
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

fn default_host() -> String {
    "0.0.0.0".to_string()
}

fn default_port() -> u16 {
    8090
}

fn default_max_concurrent() -> usize {
    128
}

fn default_billing_required() -> bool {
    true
}

fn default_monitor_interval() -> u64 {
    30
}

fn default_max_request_body_bytes() -> usize {
    16 * 1024 * 1024 // 16 MiB
}

fn default_request_timeout_secs() -> u64 {
    60
}

fn default_claim_max_retries() -> u32 {
    3
}

fn default_clock_skew_tolerance() -> u64 {
    30
}

fn default_nonce_store_path() -> Option<PathBuf> {
    Some(PathBuf::from("data/nonces.json"))
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

    /// Calculate cost for a given number of tokens (price is per-1K tokens).
    pub fn calculate_cost(&self, token_count: u32) -> u64 {
        let tokens = token_count as u64;
        // Round up: (tokens * price_per_1k + 999) / 1000
        (tokens * self.billing.price_per_1k_tokens + 999) / 1000
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn example_config_json() -> &'static str {
        r#"{
            "tangle": {
                "rpc_url": "http://localhost:8545",
                "chain_id": 31337,
                "operator_key": "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
                "tangle_core": "0x0000000000000000000000000000000000000001",
                "shielded_credits": "0x0000000000000000000000000000000000000002",
                "blueprint_id": 1,
                "service_id": null
            },
            "embedding": {
                "model": "BAAI/bge-large-en-v1.5",
                "dimensions": 1024,
                "max_sequence_length": 512
            },
            "server": {
                "host": "0.0.0.0",
                "port": 8090
            },
            "billing": {
                "price_per_1k_tokens": 1,
                "max_spend_per_request": 100000,
                "min_credit_balance": 100
            },
            "gpu": {
                "expected_gpu_count": 1,
                "min_vram_mib": 8000
            }
        }"#
    }

    #[test]
    fn test_deserialize_full_config() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert_eq!(cfg.tangle.chain_id, 31337);
        assert_eq!(cfg.embedding.model, "BAAI/bge-large-en-v1.5");
        assert_eq!(cfg.embedding.dimensions, 1024);
        assert_eq!(cfg.embedding.max_sequence_length, 512);
        assert_eq!(cfg.server.port, 8090);
        assert_eq!(cfg.billing.price_per_1k_tokens, 1);
        assert_eq!(cfg.gpu.expected_gpu_count, 1);
        assert!(cfg.tangle.service_id.is_none());
    }

    #[test]
    fn test_defaults_applied() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        assert_eq!(cfg.server.max_concurrent_requests, 128);
        assert_eq!(cfg.embedding.max_batch_size, 64);
        assert_eq!(cfg.embedding.endpoint, "http://127.0.0.1:8080");
        assert_eq!(cfg.embedding.startup_timeout_secs, 120);
        assert_eq!(cfg.gpu.monitor_interval_secs, 30);
    }

    #[test]
    fn test_calculate_cost() {
        let cfg: OperatorConfig = serde_json::from_str(example_config_json()).unwrap();
        // 1000 tokens at 1 per 1K = 1
        assert_eq!(cfg.calculate_cost(1000), 1);
        // 1 token at 1 per 1K = rounds up to 1
        assert_eq!(cfg.calculate_cost(1), 1);
        // 2500 tokens at 1 per 1K = 3 (rounds up)
        assert_eq!(cfg.calculate_cost(2500), 3);
    }

    #[test]
    fn test_missing_required_field_fails() {
        let bad = r#"{"tangle": {"rpc_url": "http://localhost:8545"}}"#;
        let result = serde_json::from_str::<OperatorConfig>(bad);
        assert!(result.is_err());
    }
}
