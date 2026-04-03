use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use tokio::sync::Semaphore;
use wiremock::{
    matchers::{method, path},
    Mock, MockServer, ResponseTemplate,
};

use embedding_inference::config::{
    BillingConfig, EmbeddingConfig, GpuConfig, OperatorConfig, ServerConfig, TangleConfig,
};
use embedding_inference::embedding::EmbeddingBackend;

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

fn test_config(backend_port: u16) -> OperatorConfig {
    OperatorConfig {
        tangle: TangleConfig {
            rpc_url: "http://localhost:8545".into(),
            chain_id: 31337,
            operator_key: "ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
                .into(),
            tangle_core: "0x0000000000000000000000000000000000000000".into(),
            shielded_credits: "0x0000000000000000000000000000000000000000".into(),
            blueprint_id: 1,
            service_id: Some(1),
        },
        embedding: EmbeddingConfig {
            model: "test-model".into(),
            dimensions: 384,
            max_sequence_length: 512,
            max_batch_size: 64,
            endpoint: format!("http://127.0.0.1:{backend_port}"),
            rerank_endpoint: None,
            rerank_model: None,
            supported_operations: vec!["embed".into(), "rerank".into()],
            hf_token: None,
            startup_timeout_secs: 10,
        },
        server: ServerConfig {
            host: "127.0.0.1".into(),
            port: 0,
            max_concurrent_requests: 128,
            max_request_body_bytes: 16 * 1024 * 1024,
            request_timeout_secs: 60,
            max_per_account_requests: 0,
        },
        billing: BillingConfig {
            required: false,
            price_per_1k_tokens: 1,
            max_spend_per_request: 100000,
            min_credit_balance: 100,
            billing_required: false,
            min_charge_amount: 0,
            claim_max_retries: 3,
            clock_skew_tolerance_secs: 30,
            max_gas_price_gwei: 0,
            nonce_store_path: None,
            payment_token_address: None,
        },
        gpu: GpuConfig {
            expected_gpu_count: 0,
            min_vram_mib: 0,
            gpu_model: None,
            monitor_interval_secs: 30,
        },
        qos: None,
    }
}

async fn start_test_server(
    backend_port: u16,
) -> (u16, tokio::sync::watch::Sender<bool>, tokio::task::JoinHandle<()>) {
    let server_port = free_port();
    let mut config = test_config(backend_port);
    config.server.port = server_port;
    let config = Arc::new(config);

    let backend = Arc::new(
        EmbeddingBackend::connect(
            format!("http://127.0.0.1:{backend_port}"),
            "test-model".into(),
        )
        .unwrap(),
    );

    let semaphore = Arc::new(Semaphore::new(128));
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);

    // Use a zero address for operator_address in tests
    let operator_address = alloy_primitives::Address::ZERO;

    let state = embedding_inference::server::AppState {
        config,
        backend,
        semaphore,
        nonce_store: Arc::new(embedding_inference::server::NonceStore::load(None)),
        active_per_account: Arc::new(RwLock::new(HashMap::new())),
        operator_address,
    };

    let handle = embedding_inference::server::start(state, shutdown_rx)
        .await
        .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    (server_port, shutdown_tx, handle)
}

// -- Tests --

#[tokio::test]
async fn test_health_check_healthy() {
    let mock = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&mock)
        .await;

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let resp = reqwest::get(format!("http://127.0.0.1:{port}/health"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "ok");
    assert_eq!(body["model"], "test-model");
    assert_eq!(body["dimensions"], 384);
}

#[tokio::test]
async fn test_health_check_unhealthy() {
    let mock = MockServer::start().await;
    // no health mock -> wiremock returns 404 -> unhealthy

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let resp = reqwest::get(format!("http://127.0.0.1:{port}/health"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 503);
}

#[tokio::test]
async fn test_embeddings_success() {
    let mock = MockServer::start().await;

    // Mock the health endpoint (needed for billing flow)
    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&mock)
        .await;

    // Mock the embedding endpoint
    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                    "index": 0
                }
            ],
            "model": "test-model",
            "usage": {
                "prompt_tokens": 5,
                "total_tokens": 5
            }
        })))
        .mount(&mock)
        .await;

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/embeddings"))
        .json(&serde_json::json!({
            "input": "hello world",
            "model": "test-model"
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["object"], "embedding");
    let embedding = body["data"][0]["embedding"].as_array().unwrap();
    assert_eq!(embedding.len(), 5);
    assert_eq!(body["usage"]["prompt_tokens"], 5);
    assert_eq!(body["usage"]["total_tokens"], 5);
}

#[tokio::test]
async fn test_embeddings_batch() {
    let mock = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&mock)
        .await;

    Mock::given(method("POST"))
        .and(path("/v1/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "data": [
                {"embedding": [0.1, 0.2, 0.3], "index": 0},
                {"embedding": [0.4, 0.5, 0.6], "index": 1}
            ],
            "model": "test-model",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        })))
        .mount(&mock)
        .await;

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/embeddings"))
        .json(&serde_json::json!({
            "input": ["hello", "world"]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["data"].as_array().unwrap().len(), 2);
}

#[tokio::test]
async fn test_embeddings_empty_input() {
    let mock = MockServer::start().await;

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/embeddings"))
        .json(&serde_json::json!({
            "input": []
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 400);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert!(body["error"]["message"].as_str().unwrap().contains("empty"));
}

#[tokio::test]
async fn test_rerank_success() {
    let mock = MockServer::start().await;

    Mock::given(method("GET"))
        .and(path("/health"))
        .respond_with(ResponseTemplate::new(200))
        .mount(&mock)
        .await;

    Mock::given(method("POST"))
        .and(path("/rerank"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
            "results": [
                {"index": 1, "relevance_score": 0.95, "document": {"text": "relevant doc"}},
                {"index": 0, "relevance_score": 0.3, "document": {"text": "less relevant"}}
            ]
        })))
        .mount(&mock)
        .await;

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let client = reqwest::Client::new();
    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/rerank"))
        .json(&serde_json::json!({
            "query": "what is rust?",
            "documents": ["less relevant", "relevant doc"]
        }))
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    let results = body["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0]["relevance_score"].as_f64().unwrap() > 0.0);
    assert!(results[0]["document"].as_str().is_some());
}

#[tokio::test]
async fn test_list_models() {
    let mock = MockServer::start().await;

    let (port, _tx, _h) = start_test_server(mock.address().port()).await;

    let resp = reqwest::get(format!("http://127.0.0.1:{port}/v1/models"))
        .await
        .unwrap();
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["object"], "list");
    assert_eq!(body["data"][0]["id"], "test-model");
}
