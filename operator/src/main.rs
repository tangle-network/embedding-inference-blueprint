use alloy_sol_types::SolValue;
use embedding_inference::config::OperatorConfig;
use embedding_inference::EmbeddingServer;

fn registration_payload(config: &OperatorConfig) -> Vec<u8> {
    let gpu_count = config.gpu.expected_gpu_count;
    let total_vram = config.gpu.min_vram_mib;
    let gpu_model = config
        .gpu
        .gpu_model
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    let endpoint = format!("http://{}:{}", config.server.host, config.server.port);

    (
        config.embedding.model.clone(),
        config.embedding.dimensions,
        config.embedding.max_sequence_length,
        gpu_count,
        total_vram,
        gpu_model,
        endpoint,
    )
        .abi_encode()
}

tangle_inference_core::run_operator!(
    OperatorConfig::load(None),
    registration_payload,
    |_| embedding_inference::router(),
    |config| EmbeddingServer { config },
    qos
);
