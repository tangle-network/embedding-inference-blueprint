use embedding_inference::embedding::{
    EmbeddingData, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
    RerankDocument, RerankRequest, RerankResponse, RerankResult,
};

#[test]
fn embedding_request_serialization() {
    let req = EmbeddingRequest {
        input: vec!["hello world".to_string(), "test input".to_string()],
        model: "BAAI/bge-large-en-v1.5".to_string(),
    };

    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("hello world"));
    assert!(json.contains("BAAI/bge-large-en-v1.5"));

    let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed["input"].as_array().unwrap().len(), 2);
}

#[test]
fn embedding_response_deserialization() {
    let json = r#"{
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
            {"object": "embedding", "embedding": [0.4, 0.5, 0.6], "index": 1}
        ],
        "model": "BAAI/bge-large-en-v1.5",
        "usage": {"prompt_tokens": 10, "total_tokens": 10}
    }"#;

    let resp: EmbeddingResponse = serde_json::from_str(json).unwrap();

    assert_eq!(resp.object.as_deref(), Some("list"));
    assert_eq!(resp.data.len(), 2);
    assert_eq!(resp.data[0].embedding, vec![0.1, 0.2, 0.3]);
    assert_eq!(resp.data[0].index, 0);
    assert_eq!(resp.data[1].index, 1);
    assert_eq!(resp.model.as_deref(), Some("BAAI/bge-large-en-v1.5"));
    assert_eq!(resp.usage.as_ref().unwrap().prompt_tokens, 10);
    assert_eq!(resp.usage.as_ref().unwrap().total_tokens, 10);
}

#[test]
fn embedding_response_roundtrip() {
    let resp = EmbeddingResponse {
        object: Some("list".to_string()),
        data: vec![EmbeddingData {
            object: Some("embedding".to_string()),
            embedding: vec![0.1, 0.2, 0.3, 0.4],
            index: 0,
        }],
        model: Some("test-model".to_string()),
        usage: Some(EmbeddingUsage {
            prompt_tokens: 5,
            total_tokens: 5,
        }),
    };

    let json = serde_json::to_string(&resp).unwrap();
    let deserialized: EmbeddingResponse = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.data.len(), 1);
    assert_eq!(deserialized.data[0].embedding.len(), 4);
    assert_eq!(deserialized.usage.as_ref().unwrap().prompt_tokens, 5);
}

#[test]
fn embedding_response_minimal_fields() {
    // Backend might return minimal response without optional fields
    let json = r#"{
        "data": [{"embedding": [1.0, 2.0], "index": 0}]
    }"#;

    let resp: EmbeddingResponse = serde_json::from_str(json).unwrap();
    assert!(resp.object.is_none());
    assert!(resp.model.is_none());
    assert!(resp.usage.is_none());
    assert_eq!(resp.data[0].embedding, vec![1.0, 2.0]);
}

#[test]
fn rerank_request_serialization() {
    let req = RerankRequest {
        model: "BAAI/bge-reranker-v2-m3".to_string(),
        query: "what is machine learning?".to_string(),
        documents: vec![
            "ML is a subset of AI".to_string(),
            "Cooking recipes for pasta".to_string(),
        ],
        top_n: Some(1),
    };

    let json = serde_json::to_string(&req).unwrap();
    assert!(json.contains("bge-reranker"));
    assert!(json.contains("top_n"));
}

#[test]
fn rerank_request_without_top_n() {
    let req = RerankRequest {
        model: "model".to_string(),
        query: "query".to_string(),
        documents: vec!["doc".to_string()],
        top_n: None,
    };

    let json = serde_json::to_string(&req).unwrap();
    // top_n should be skipped when None
    assert!(!json.contains("top_n"));
}

#[test]
fn rerank_response_deserialization() {
    let json = r#"{
        "results": [
            {"index": 0, "relevance_score": 0.95, "document": {"text": "ML is a subset of AI"}},
            {"index": 1, "relevance_score": 0.12}
        ]
    }"#;

    let resp: RerankResponse = serde_json::from_str(json).unwrap();

    assert_eq!(resp.results.len(), 2);
    assert_eq!(resp.results[0].index, 0);
    assert!((resp.results[0].relevance_score - 0.95).abs() < f64::EPSILON);
    assert_eq!(
        resp.results[0].document.as_ref().unwrap().text,
        "ML is a subset of AI"
    );
    assert!(resp.results[1].document.is_none());
}

#[test]
fn rerank_result_roundtrip() {
    let result = RerankResult {
        index: 2,
        relevance_score: 0.87,
        document: Some(RerankDocument {
            text: "relevant document".to_string(),
        }),
    };

    let json = serde_json::to_string(&result).unwrap();
    let deserialized: RerankResult = serde_json::from_str(&json).unwrap();

    assert_eq!(deserialized.index, 2);
    assert!((deserialized.relevance_score - 0.87).abs() < f64::EPSILON);
}
