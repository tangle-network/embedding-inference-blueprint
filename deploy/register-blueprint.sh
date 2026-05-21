#!/usr/bin/env bash
# Register the embedding-inference blueprint on Tangle.
#
# Two-stage flow:
#   1. Deploy EmbeddingBSM behind an ERC1967 UUPS proxy.
#        a. `forge create EmbeddingBSM` — deploys the implementation.
#        b. `forge create ERC1967Proxy <impl> <encodedInitialize>` — deploys the
#           proxy and atomically calls `initialize(tsUSD)`.
#      The proxy address is what gets registered as the blueprint manager.
#   2. `cargo tangle blueprint deploy tangle` — patches `manager` in a temp
#      copy of `deploy/definition.json` with the proxy address, then registers
#      the blueprint via `Tangle.createBlueprint()` on the target network.
#
# Prerequisites:
#   - forge (Foundry) installed and on PATH
#   - cargo-tangle CLI installed (`cargo install cargo-tangle`)
#   - jq installed
#   - Deployer wallet funded on the target network
#   - Keystore directory with the deployer key (defaults to ./keystore)
#
# Usage (Base Sepolia, against the deployed Tangle protocol):
#
#   export PRIVATE_KEY=0x...
#   export RPC_URL=https://sepolia.base.org
#   export WS_URL=wss://base-sepolia-rpc.publicnode.com
#   export TANGLE_CORE=0xC9b0716a187072be0f38A5D972392C6479b9Cfe3
#   export TSUSD_ADDRESS=0x036CbD53842c5426634e7929541eC2318f3dCF7e  # USDC sepolia (default)
#   export KEYSTORE_PATH=./keystore
#   ./deploy/register-blueprint.sh
#
# Optional:
#   BSM_ADDRESS  — skip the forge create steps if the proxy is already deployed
#                  (definition.json gets patched with this address instead).

set -euo pipefail

: "${RPC_URL:?Set RPC_URL}"
: "${PRIVATE_KEY:?Set PRIVATE_KEY}"
: "${TANGLE_CORE:?Set TANGLE_CORE}"
: "${WS_URL:?Set WS_URL (ws://… or wss://…)}"
: "${KEYSTORE_PATH:=./keystore}"

# Default payment token: USDC on Base Sepolia. EmbeddingBSM accepts this as
# the tsUSD wrapper for shielded billing.
: "${TSUSD_ADDRESS:=0x036CbD53842c5426634e7929541eC2318f3dCF7e}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONTRACTS_DIR="$REPO_ROOT/contracts"
DEFINITION_FILE="$REPO_ROOT/deploy/definition.json"

echo "=== Embedding-Inference Blueprint Registration ==="
echo "Network:     $(cast chain-id --rpc-url "$RPC_URL")"
echo "Deployer:    $(cast wallet address --private-key "$PRIVATE_KEY")"
echo "Tangle Core: $TANGLE_CORE"
echo "tsUSD:       $TSUSD_ADDRESS"
echo "Definition:  $DEFINITION_FILE"
echo ""

# Stage 1 — Deploy EmbeddingBSM (impl + UUPS proxy) unless BSM_ADDRESS is set.
# EmbeddingBSM is upgradeable: the constructor calls `_disableInitializers()`
# on the implementation, and the proxy's `initialize(tsUSD)` is invoked at
# construction time via ERC1967Proxy's `_data` argument.
if [ -z "${BSM_ADDRESS:-}" ]; then
    echo "Stage 1a: deploying EmbeddingBSM implementation …"
    # `forge create --json` interleaves compile progress with JSON on stdout,
    # so jq parsing breaks intermittently. Grep the human-readable
    # "Deployed to:" line instead — it's stable across forge versions and
    # cache states.
    IMPL_ADDRESS=$(forge create \
        --root "$CONTRACTS_DIR" \
        --rpc-url "$RPC_URL" \
        --private-key "$PRIVATE_KEY" \
        --broadcast \
        "$CONTRACTS_DIR/src/EmbeddingBSM.sol:EmbeddingBSM" \
        --json 2>&1 | grep -oE 'Deployed to: 0x[a-fA-F0-9]{40}' | tail -1 | awk '{print $3}')
    echo "$IMPL_ADDRESS" | grep -qE '^0x[a-fA-F0-9]{40}$' \
        || { echo "failed to extract EmbeddingBSM impl address from forge create output"; exit 1; }
    echo "EmbeddingBSM impl deployed at: $IMPL_ADDRESS"

    echo "Stage 1b: deploying ERC1967Proxy and invoking initialize(tsUSD) …"
    # `initialize(address)` selector = 0xc4d66de8; pad tsUSD to 32 bytes.
    INIT_CALLDATA="0xc4d66de8000000000000000000000000${TSUSD_ADDRESS#0x}"
    BSM_ADDRESS=$(forge create \
        --root "$CONTRACTS_DIR" \
        --rpc-url "$RPC_URL" \
        --private-key "$PRIVATE_KEY" \
        --broadcast \
        "$CONTRACTS_DIR/dependencies/@openzeppelin-contracts-5.1.0/proxy/ERC1967/ERC1967Proxy.sol:ERC1967Proxy" \
        --constructor-args "$IMPL_ADDRESS" "$INIT_CALLDATA" \
        --json 2>&1 | grep -oE 'Deployed to: 0x[a-fA-F0-9]{40}' | tail -1 | awk '{print $3}')
    echo "$BSM_ADDRESS" | grep -qE '^0x[a-fA-F0-9]{40}$' \
        || { echo "failed to extract ERC1967Proxy address from forge create output"; exit 1; }
    echo "EmbeddingBSM proxy deployed at: $BSM_ADDRESS"
else
    echo "Stage 1 skipped — reusing existing BSM proxy at $BSM_ADDRESS"
fi
echo ""

# Stage 2 — Patch deploy/definition.json with the BSM proxy address and call
# cargo-tangle's canonical deploy flow. The patched file is written to a temp
# path so the in-tree definition stays untouched (its `manager: 0x0…0` is
# the template).
PATCHED_DEFINITION=$(mktemp --suffix=-embedding-blueprint.json)
trap 'rm -f "$PATCHED_DEFINITION"' EXIT
jq --arg mgr "$BSM_ADDRESS" '.manager = $mgr' "$DEFINITION_FILE" > "$PATCHED_DEFINITION"

echo "Stage 2: cargo tangle blueprint deploy tangle …"
cargo tangle blueprint deploy tangle \
    --network testnet \
    --definition "$PATCHED_DEFINITION" \
    --http-rpc-url "$RPC_URL" \
    --ws-rpc-url "$WS_URL" \
    --tangle-contract "$TANGLE_CORE" \
    --keystore-path "$KEYSTORE_PATH"

echo ""
echo "=== Blueprint registered ==="
echo "EmbeddingBSM proxy: $BSM_ADDRESS"
echo "(blueprint ID is logged by cargo-tangle above)"
