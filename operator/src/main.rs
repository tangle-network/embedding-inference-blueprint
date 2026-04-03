use blueprint_sdk::std::sync::Arc;

use alloy_sol_types::SolValue;
use blueprint_sdk::contexts::tangle::TangleClientContext;
use blueprint_sdk::runner::config::BlueprintEnvironment;
use blueprint_sdk::runner::tangle::config::TangleConfig;
use blueprint_sdk::runner::BlueprintRunner;
use blueprint_sdk::tangle::{TangleConsumer, TangleProducer};

use embedding_inference::config::OperatorConfig;
use embedding_inference::health;
use embedding_inference::EmbeddingServer;

fn setup_log() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::from_default_env();
    fmt().with_env_filter(filter).init();
}

/// Build ABI-encoded registration payload for EmbeddingBSM.onRegister.
/// Format: abi.encode(string model, uint32 dimensions, uint32 maxSeqLen, uint32 gpuCount, uint32 totalVramMib, string gpuModel, string endpoint)
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

#[tokio::main]
#[allow(clippy::result_large_err)]
async fn main() -> Result<(), blueprint_sdk::Error> {
    setup_log();

    let config = OperatorConfig::load(None)
        .map_err(|e| blueprint_sdk::Error::Other(format!("config load failed: {e}")))?;
    let config = Arc::new(config);

    let env = BlueprintEnvironment::load()?;

    // Registration mode: emit registration inputs and exit
    if env.registration_mode() {
        let payload = registration_payload(&config);
        let output_path = env.registration_output_path();
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;
        }
        std::fs::write(&output_path, &payload)
            .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;
        tracing::info!(
            path = %output_path.display(),
            model = %config.embedding.model,
            dimensions = config.embedding.dimensions,
            "Registration payload saved"
        );
        return Ok(());
    }

    // Check GPU availability (non-fatal)
    match health::detect_gpus().await {
        Ok(gpus) => {
            tracing::info!(count = gpus.len(), "detected GPUs");
            for gpu in &gpus {
                tracing::info!(name = %gpu.name, vram_mib = gpu.memory_total_mib, "GPU");
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "GPU detection failed -- running in CPU mode");
        }
    }

    let tangle_client = env
        .tangle_client()
        .await
        .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?;

    let service_id = env
        .protocol_settings
        .tangle()
        .map_err(|e| blueprint_sdk::Error::Other(e.to_string()))?
        .service_id
        .ok_or_else(|| blueprint_sdk::Error::Other("No service ID configured".to_string()))?;

    let tangle_producer = TangleProducer::new(tangle_client.clone(), service_id);
    let tangle_consumer = TangleConsumer::new(tangle_client.clone());

    // QoS heartbeat
    let qos_enabled = config
        .qos
        .as_ref()
        .map(|q| q.heartbeat_interval_secs > 0)
        .unwrap_or(false);
    if qos_enabled {
        match embedding_inference::qos::start_heartbeat(config.clone()).await {
            Ok(_handle) => {
                let interval = config.qos.as_ref().unwrap().heartbeat_interval_secs;
                tracing::info!(interval_secs = interval, "QoS heartbeat started");
            }
            Err(e) => {
                tracing::warn!(error = %e, "QoS heartbeat failed to start (disabled)");
            }
        }
    } else {
        tracing::info!("QoS heartbeat disabled");
    }

    // Background service: embedding backend + HTTP server
    let embedding_server = EmbeddingServer {
        config: config.clone(),
    };

    BlueprintRunner::builder(TangleConfig::default(), env)
        .router(embedding_inference::router())
        .producer(tangle_producer)
        .consumer(tangle_consumer)
        .background_service(embedding_server)
        .run()
        .await?;

    Ok(())
}
