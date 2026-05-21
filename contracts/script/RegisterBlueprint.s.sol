// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { Script, console2 } from "forge-std/Script.sol";
import { ERC1967Proxy } from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";
import { Types } from "tnt-core/libraries/Types.sol";
import { EmbeddingBSM } from "../src/EmbeddingBSM.sol";

/// @notice Minimal interface for Tangle blueprint registration.
interface ITangle {
    function createBlueprint(Types.BlueprintDefinition calldata def) external returns (uint64);
}

/// @title RegisterBlueprint
/// @notice Deploys EmbeddingBSM (impl + UUPS proxy + initialize) and registers
///         the embedding-inference blueprint on Tangle in a single broadcast.
/// @dev    Run via: `forge script contracts/script/RegisterBlueprint.s.sol
///         --rpc-url $RPC_URL --broadcast --slow`
///
///         Mirrors the proven single-broadcast pattern used by sibling
///         blueprints (deploy BSM impl + UUPS proxy + initialize + Tangle
///         createBlueprint in one transaction sequence).
contract RegisterBlueprint is Script {
    // ─────────────────────────────────────────────────────────────────────────
    // Defaults — overridable via env vars for non-anvil chains.
    // ─────────────────────────────────────────────────────────────────────────

    // Anvil well-known deployer key (default when no PRIVATE_KEY env is set).
    uint256 constant DEFAULT_DEPLOYER_KEY =
        0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80;

    // Tangle protocol address on a LocalTestnet anvil snapshot. For real
    // chains (Base Sepolia, mainnet) pass TANGLE_CORE via env.
    address constant DEFAULT_TANGLE = 0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9;

    // USDC on Base Sepolia. EmbeddingBSM accepts this as the tsUSD wrapper
    // under the shielded billing flow. For other networks pass TSUSD_ADDRESS
    // via env.
    address constant DEFAULT_TSUSD = 0x036CbD53842c5426634e7929541eC2318f3dCF7e;

    function run() external {
        uint256 deployerKey = vm.envOr("PRIVATE_KEY", DEFAULT_DEPLOYER_KEY);
        address tangleAddr = vm.envOr("TANGLE_CORE", DEFAULT_TANGLE);
        address tsUSD = vm.envOr("TSUSD_ADDRESS", DEFAULT_TSUSD);

        ITangle tangle = ITangle(tangleAddr);

        vm.startBroadcast(deployerKey);

        // ── Deploy EmbeddingBSM (UUPS impl + proxy + initialize) ────────────
        EmbeddingBSM impl = new EmbeddingBSM();
        ERC1967Proxy proxy = new ERC1967Proxy(
            address(impl),
            abi.encodeCall(EmbeddingBSM.initialize, (tsUSD))
        );
        EmbeddingBSM bsm = EmbeddingBSM(payable(address(proxy)));

        // ── Register on Tangle ──────────────────────────────────────────────
        uint64 blueprintId = tangle.createBlueprint(_buildDefinition(address(bsm)));

        vm.stopBroadcast();

        // ── Output for bash wrapper parsing ─────────────────────────────────
        console2.log("DEPLOY_EMBEDDING_BSM_IMPL=%s", vm.toString(address(impl)));
        console2.log("DEPLOY_EMBEDDING_BSM_PROXY=%s", vm.toString(address(bsm)));
        console2.log("DEPLOY_EMBEDDING_BLUEPRINT_ID=%s", vm.toString(blueprintId));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Blueprint Definition builder
    // ═════════════════════════════════════════════════════════════════════════

    function _buildDefinition(address manager) internal pure returns (Types.BlueprintDefinition memory def) {
        def.metadataUri = "https://github.com/tangle-network/embedding-inference-blueprint";
        // metadataHash is a digest of the canonical metadata JSON. Until that
        // payload is pinned via IPFS, derive it from the metadataUri so the
        // value is deterministic + traceable.
        def.metadataHash = keccak256(bytes(def.metadataUri));
        def.manager = manager;
        def.masterManagerRevision = 0;
        def.hasConfig = true;

        // Event-driven pricing: operators are paid per embedding job rather
        // than on a fixed subscription cadence. `deploy/definition.json`
        // declares `membership: Dynamic`, `min_operators: 1`,
        // `max_operators: 0` — that maps to dynamic membership + event-driven
        // pricing here.
        def.config = Types.BlueprintConfig({
            membership: Types.MembershipModel.Dynamic,
            pricing: Types.PricingModel.EventDriven,
            minOperators: 1,
            maxOperators: 0, // unbounded
            subscriptionRate: 0,
            subscriptionInterval: 0,
            eventRate: 0 // operators negotiate price per call via RFQ
        });

        def.metadata = Types.BlueprintMetadata({
            name: "Embedding Inference Blueprint",
            description: "Text embedding operator (dense vector generation) via TEI on Tangle",
            author: "Tangle Network",
            category: "AI/Embeddings",
            codeRepository: "https://github.com/tangle-network/embedding-inference-blueprint",
            logo: "",
            website: "https://tangle.tools",
            license: "MIT OR Apache-2.0",
            profilingData: "{\"execution_profile\":{\"gpu\":{\"policy\":\"preferred\",\"min_count\":1,\"min_vram_gb\":4}}}"
        });

        def.jobs = _buildJobs();

        def.registrationSchema = "";
        def.requestSchema = "";

        def.sources = new Types.BlueprintSource[](1);
        Types.BlueprintBinary[] memory bins = new Types.BlueprintBinary[](1);
        bins[0] = Types.BlueprintBinary({
            arch: Types.BlueprintArchitecture.Amd64,
            os: Types.BlueprintOperatingSystem.Linux,
            name: "embedding-operator",
            sha256: bytes32(uint256(0xdeadbeef))
        });
        def.sources[0] = Types.BlueprintSource({
            kind: Types.BlueprintSourceKind.Native,
            container: Types.ImageRegistrySource("", "", ""),
            wasm: Types.WasmSource(Types.WasmRuntime.Unknown, Types.BlueprintFetcherKind.None, "", ""),
            native: Types.NativeSource(
                Types.BlueprintFetcherKind.None,
                "file:///target/release/embedding-operator",
                "./target/release/embedding-operator"
            ),
            testing: Types.TestingSource("embedding-inference", "embedding-operator", "."),
            binaries: bins
        });

        // `deploy/definition.json` declares both Dynamic and Fixed in
        // `supported_memberships`; surface that here so off-chain consumers
        // can pick either.
        def.supportedMemberships = new Types.MembershipModel[](2);
        def.supportedMemberships[0] = Types.MembershipModel.Dynamic;
        def.supportedMemberships[1] = Types.MembershipModel.Fixed;
    }

    function _buildJobs() internal pure returns (Types.JobDefinition[] memory jobs) {
        jobs = new Types.JobDefinition[](1);
        // Job 0: embed
        //   params:  (string model, string input)
        //   result:  (bytes vectors, uint32 totalTokens)
        // The Rust operator enforces these shapes; on-chain schemas are kept
        // empty to match the pattern used by sibling blueprints where
        // params/result types live with the running operator, not the
        // Blueprint registry. Future PR can introduce hex-encoded schemas via
        // tnt-core's SchemaLib once that surface stabilizes across repos.
        jobs[0] = Types.JobDefinition({
            name: "embed",
            description: "Generate dense embedding vectors for text inputs via the TEI backend",
            metadataUri: "",
            paramsSchema: "",
            resultSchema: ""
        });
    }
}
