#!/usr/bin/env bash
# Register the embedding-inference blueprint on Tangle.
#
# Single-shot flow: deploys EmbeddingBSM (impl + UUPS proxy + initialize)
# AND calls Tangle.createBlueprint in the same broadcast via
# `contracts/script/Deploy.s.sol`. This replaces the prior
# two-stage `forge create` + `cargo tangle blueprint deploy` flow.
#
# Prerequisites:
#   - forge installed
#   - Deployer wallet funded on the target network
#
# Usage (Base Sepolia, against the already-deployed Tangle protocol):
#
#   export PRIVATE_KEY=0x...
#   export RPC_URL=https://sepolia.base.org
#   export TANGLE_CORE=0xC9b0716a187072be0f38A5D972392C6479b9Cfe3
#   export TSUSD_ADDRESS=0x036CbD53842c5426634e7929541eC2318f3dCF7e   # USDC sepolia (default)
#   ./deploy/register-blueprint.sh
#
# Local anvil (LocalTestnet snapshot):
#
#   export RPC_URL=http://127.0.0.1:8545
#   ./deploy/register-blueprint.sh   # uses anvil deployer key + Tangle/USDC defaults
#
# Optional overrides for the operator-registration calldata emitted at the
# end of the run:
#   MODEL              embedding model name (default: BAAI/bge-large-en-v1.5)
#   DIMENSIONS         embedding output dimensions (default: 1024)
#   MAX_SEQ_LEN        max input sequence length in tokens (default: 512)
#   GPU_COUNT          GPUs the operator exposes (default: 1)
#   TOTAL_VRAM         total VRAM in MiB (default: 24000)
#   GPU_MODEL          GPU model string (default: NVIDIA L4)
#   ENDPOINT           operator HTTP endpoint (default: https://your-operator.example.com)
#
# Outputs (parsed by deployment scripts, do not change without coordinating):
#   DEPLOY_EMBEDDING_BSM_IMPL=<address>
#   DEPLOY_EMBEDDING_BSM_PROXY=<address>
#   DEPLOY_EMBEDDING_BLUEPRINT_ID=<u64>

set -euo pipefail

: "${RPC_URL:?Set RPC_URL}"
: "${PRIVATE_KEY:?Set PRIVATE_KEY}"
TANGLE_CORE="${TANGLE_CORE:-0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9}"

MODEL="${MODEL:-BAAI/bge-large-en-v1.5}"
DIMENSIONS="${DIMENSIONS:-1024}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
GPU_COUNT="${GPU_COUNT:-1}"
TOTAL_VRAM="${TOTAL_VRAM:-24000}"
GPU_MODEL="${GPU_MODEL:-NVIDIA L4}"
ENDPOINT="${ENDPOINT:-https://your-operator.example.com}"

echo "=== Embedding-Inference Blueprint Registration ==="
echo "Network:     $(cast chain-id --rpc-url "$RPC_URL")"
echo "Tangle Core: ${TANGLE_CORE:-<default from RegisterBlueprint.s.sol>}"
echo "tsUSD:       ${TSUSD_ADDRESS:-<default USDC sepolia>}"
echo "Model:       $MODEL ($DIMENSIONS-dim, max_seq=$MAX_SEQ_LEN)"
echo "GPUs:        $GPU_COUNT x $GPU_MODEL ($TOTAL_VRAM MiB)"
echo "Endpoint:    $ENDPOINT"
echo ""

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CORE_DIR="$(cargo metadata --manifest-path "$ROOT_DIR/Cargo.toml" --format-version 1 --locked | python3 -c 'import json,sys,pathlib; print(pathlib.Path(next(p["manifest_path"] for p in json.load(sys.stdin)["packages"] if p["name"] == "tangle-inference-core")).parent)')"
cd "$(dirname "$0")/../contracts"

# Deploy BSM (impl + proxy + initialize) AND register the blueprint in one
# forge-script broadcast.
DEPLOY_OUTPUT=$(PRIVATE_KEY="$PRIVATE_KEY" \
    TANGLE_CORE="${TANGLE_CORE:-}" \
    TSUSD_ADDRESS="${TSUSD_ADDRESS:-0x036CbD53842c5426634e7929541eC2318f3dCF7e}" \
    forge script script/Deploy.s.sol:Deploy \
        --remappings "tangle-inference-core/=$CORE_DIR/contracts/" \
        --rpc-url "$RPC_URL" \
        --broadcast --slow)

echo "$DEPLOY_OUTPUT"

# Extract the BSM proxy address + blueprint ID for downstream scripts.
BSM_ADDRESS=$(echo "$DEPLOY_OUTPUT" | grep -oE 'DEPLOY_EMBEDDING_BSM_PROXY=0x[0-9a-fA-F]+' | tail -1 | cut -d= -f2)
BLUEPRINT_ID=$(echo "$DEPLOY_OUTPUT" | grep -oE 'DEPLOY_EMBEDDING_BLUEPRINT_ID=[0-9]+' | tail -1 | cut -d= -f2)

if [ -z "$BSM_ADDRESS" ] || [ -z "$BLUEPRINT_ID" ]; then
    echo "ERROR: failed to extract addresses from forge output"
    exit 1
fi

echo ""
echo "=== Blueprint registered ==="
echo "Blueprint ID:        $BLUEPRINT_ID"
echo "EmbeddingBSM proxy:  $BSM_ADDRESS"
echo ""

# Operator registration is a separate step (per-operator). Encode the
# registration inputs so the operator can call Tangle.registerOperator
# with the right calldata. The signature mirrors EmbeddingBSM.onRegister's
# abi.decode shape: (string,uint32,uint32,uint32,uint32,string,string).
REG_INPUTS=$(cast abi-encode \
    "f(string,uint32,uint32,uint32,uint32,string,string)" \
    "$MODEL" "$DIMENSIONS" "$MAX_SEQ_LEN" "$GPU_COUNT" "$TOTAL_VRAM" "$GPU_MODEL" "$ENDPOINT")

echo "Operator registration inputs (use these to register an operator):"
echo "  $REG_INPUTS"
echo ""
