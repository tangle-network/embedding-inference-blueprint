//! Full lifecycle test -- embedding job through real handler + wiremock backend.

use anyhow::{Result, ensure};
use wiremock::{MockServer, Mock, ResponseTemplate, matchers::{method, path}};
use embedding_inference::EmbeddingRequest;

#[tokio::test]
async fn test_embed_direct_with_wiremock() -> Result<()> {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                    "index": 0
                },
                {
                    "object": "embedding",
                    "embedding": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                    "index": 1
                }
            ],
            "model": "bge-large-en-v1.5",
            "usage": {
                "prompt_tokens": 12,
                "total_tokens": 12
            }
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    embedding_inference::init_for_testing(&mock_server.uri(), "bge-large-en-v1.5");

    let request = EmbeddingRequest {
        inputs: vec![
            "Hello world".to_string(),
            "Goodbye world".to_string(),
        ],
    };

    let result = embedding_inference::embed_direct(&request).await;

    match result {
        Ok(embed_result) => {
            ensure!(embed_result.count == 2, "expected 2 embeddings, got {}", embed_result.count);
            ensure!(embed_result.totalTokens == 12, "expected 12 tokens, got {}", embed_result.totalTokens);
            ensure!(embed_result.dimensions == 8, "expected 8 dimensions, got {}", embed_result.dimensions);
        }
        Err(e) => panic!("Embedding failed: {e}"),
    }

    mock_server.verify().await;

    Ok(())
}
