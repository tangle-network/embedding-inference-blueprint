//! Embedding backend client.
//!
//! Wraps HuggingFace TEI (text-embeddings-inference) or any OpenAI-compatible
//! embedding server. The backend must accept:
//!   POST /v1/embeddings  { "input": ["text", ...], "model": "..." }
//! and return:
//!   { "data": [{"embedding": [...], "index": 0}], "usage": {"prompt_tokens": N, "total_tokens": N} }

use blueprint_sdk::std::sync::Arc;
use blueprint_sdk::std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config::OperatorConfig;

/// Client for communicating with the embedding backend.
pub struct EmbeddingClient {
    client: reqwest::Client,
    endpoint: String,
    model: String,
    /// Separate endpoint for reranking; falls back to self.endpoint.
    rerank_endpoint: Option<String>,
}

/// Request body sent to the embedding backend.
#[derive(Debug, Serialize)]
pub struct EmbeddingRequest {
    pub input: Vec<String>,
    pub model: String,
}

/// Response from the embedding backend.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct EmbeddingResponse {
    pub object: Option<String>,
    pub data: Vec<EmbeddingData>,
    pub model: Option<String>,
    pub usage: Option<EmbeddingUsage>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct EmbeddingData {
    pub object: Option<String>,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

/// Request body sent to the reranking backend.
#[derive(Debug, Serialize)]
pub struct RerankRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_n: Option<usize>,
}

/// Response from the reranking backend (Cohere-compatible).
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct RerankResponse {
    pub results: Vec<RerankResult>,
}

/// A single scored document in a rerank response.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct RerankResult {
    pub index: usize,
    pub relevance_score: f64,
    #[serde(default)]
    pub document: Option<RerankDocument>,
}

/// Document wrapper returned by some reranking backends.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct RerankDocument {
    pub text: String,
}

impl EmbeddingClient {
    /// Create a new backend client from config.
    pub fn new(config: &OperatorConfig) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(config.embedding.startup_timeout_secs))
            .build()?;

        let endpoint = config.embedding.endpoint.clone();
        let model = config.embedding.model.clone();
        let rerank_endpoint = config.embedding.rerank_endpoint.clone();

        Ok(Self {
            client,
            endpoint,
            model,
            rerank_endpoint,
        })
    }

    /// Connect to an already-running embedding server (for external management).
    pub fn connect(endpoint: String, model: String) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()?;

        Ok(Self {
            client,
            endpoint,
            model,
            rerank_endpoint: None,
        })
    }

    /// Check if the embedding backend is healthy.
    pub async fn is_healthy(&self) -> bool {
        let url = format!("{}/health", self.endpoint);
        matches!(
            self.client
                .get(&url)
                .timeout(Duration::from_secs(5))
                .send()
                .await,
            Ok(r) if r.status().is_success()
        )
    }

    /// Wait for the embedding backend to become ready.
    pub async fn wait_ready(&self, config: &OperatorConfig) -> anyhow::Result<()> {
        let url = format!("{}/health", self.endpoint);
        let timeout = Duration::from_secs(config.embedding.startup_timeout_secs);
        let start = std::time::Instant::now();

        loop {
            if start.elapsed() > timeout {
                anyhow::bail!(
                    "embedding backend failed to become ready within {}s",
                    config.embedding.startup_timeout_secs
                );
            }

            match self.client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => return Ok(()),
                _ => {
                    tokio::time::sleep(Duration::from_secs(2)).await;
                }
            }
        }
    }

    /// Generate embeddings for a batch of inputs.
    pub async fn embed(&self, inputs: Vec<String>) -> anyhow::Result<EmbeddingResponse> {
        let url = format!("{}/v1/embeddings", self.endpoint);

        let body = EmbeddingRequest {
            input: inputs,
            model: self.model.clone(),
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<EmbeddingResponse>()
            .await?;

        Ok(resp)
    }

    /// Generate embeddings with a specific model override.
    pub async fn embed_with_model(
        &self,
        inputs: Vec<String>,
        model: &str,
    ) -> anyhow::Result<EmbeddingResponse> {
        let url = format!("{}/v1/embeddings", self.endpoint);

        let body = EmbeddingRequest {
            input: inputs,
            model: model.to_string(),
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<EmbeddingResponse>()
            .await?;

        Ok(resp)
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// Resolved rerank endpoint: RERANK_ENDPOINT if set, else embedding endpoint.
    pub fn rerank_base_url(&self) -> &str {
        self.rerank_endpoint.as_deref().unwrap_or(&self.endpoint)
    }

    /// Rerank documents against a query using a cross-encoder model.
    pub async fn rerank(
        &self,
        query: String,
        documents: Vec<String>,
        model: &str,
        top_n: Option<usize>,
    ) -> anyhow::Result<RerankResponse> {
        let base = self.rerank_base_url();
        let url = format!("{base}/rerank");

        let body = RerankRequest {
            model: model.to_string(),
            query,
            documents,
            top_n,
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await?
            .error_for_status()?
            .json::<RerankResponse>()
            .await?;

        Ok(resp)
    }
}

/// Create a shared EmbeddingClient from config.
pub fn create_client(config: Arc<OperatorConfig>) -> anyhow::Result<Arc<EmbeddingClient>> {
    let client = EmbeddingClient::new(&config)?;
    Ok(Arc::new(client))
}
