![Tangle Network Banner](https://raw.githubusercontent.com/tangle-network/tangle/refs/heads/main/assets/Tangle%20%20Banner.png)

<h1 align="center">Embedding Blueprint</h1>

<p align="center"><em>Decentralized text embeddings on <a href="https://tangle.tools">Tangle</a> — operators serve embedding models via HuggingFace TEI or any OpenAI-compatible endpoint.</em></p>

<p align="center">
  <a href="https://discord.com/invite/cv8EfJu3Tn"><img src="https://img.shields.io/discord/833784453251596298?label=Discord" alt="Discord"></a>
  <a href="https://t.me/tanglenet"><img src="https://img.shields.io/endpoint?color=neon&url=https%3A%2F%2Ftg.sumanjay.workers.dev%2Ftanglenet" alt="Telegram"></a>
</p>

## Overview

A Tangle Blueprint enabling operators to serve text embedding models with anonymous payments through shielded credits. High-volume, low-cost embeddings for RAG, search, and classification workloads.

**Dual payment paths:**
- **On-chain jobs** via TangleProducer — verifiable results on Tangle
- **x402 HTTP** — fast private embeddings at `/v1/embeddings`

OpenAI Embeddings API compatible. Built with [Blueprint SDK](https://github.com/tangle-network/blueprint) with TEE support.

## Components

| Component | Language | Description |
|-----------|----------|-------------|
| `operator/` | Rust | Operator binary — wraps TEI or embedding endpoint, HTTP server, SpendAuth billing |
| `contracts/` | Solidity | EmbeddingBSM — dimension validation, per-1K-token pricing |

## Supported Models

Any model compatible with HuggingFace [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference):
- BGE (large, base, small)
- E5 (large, base, small)
- Jina Embeddings v3
- GTE (Qwen)
- Nomic Embed

## Pricing

Per-1K-tokens. Operators compete on price — typically 5-10x cheaper than OpenAI's embedding API.

## TEE Support

Add `features = ["tee"]` to `blueprint-sdk` in Cargo.toml. The `TeeLayer` middleware transparently attaches attestation metadata when running in a Confidential VM. Passes through when no TEE is configured.

## Quick Start

```bash
# Start TEI (example with BGE-large)
docker run -p 8080:80 ghcr.io/huggingface/text-embeddings-inference:latest \
  --model-id BAAI/bge-large-en-v1.5

# Configure operator
cp config/operator.example.toml config/operator.toml

# Run operator
EMBEDDING_ENDPOINT=http://localhost:8080 cargo run --release
```

## Related Repos

- [Blueprint SDK](https://github.com/tangle-network/blueprint) — framework for building Blueprints
- [vLLM Inference Blueprint](https://github.com/tangle-network/vllm-inference-blueprint) — text inference
- [Voice Inference Blueprint](https://github.com/tangle-network/voice-inference-blueprint) — TTS/STT
- [Image Generation Blueprint](https://github.com/tangle-network/image-gen-inference-blueprint) — image generation
- [Video Generation Blueprint](https://github.com/tangle-network/video-gen-inference-blueprint) — video generation
