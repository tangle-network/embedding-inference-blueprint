// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import {Script, console2} from "forge-std/Script.sol";
import {EmbeddingBSM} from "../src/EmbeddingBSM.sol";
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";

/// Model-specific manager deployment; cargo-tangle registers the definition.
contract Deploy is Script {
    function run() external returns (address manager) {
        address registry = vm.envAddress("TANGLE_CORE");
        require(registry.code.length != 0, "TANGLE_CORE has no code");
        address token = vm.envAddress("PAYMENT_TOKEN");
        // The operator config is the one source of the served model, its dimensions, price and VRAM floor.
        string memory config = vm.readFile(vm.envOr("MODEL_CONFIG", string("../deploy/config.example.json")));
        vm.startBroadcast();
        EmbeddingBSM implementation = new EmbeddingBSM();
        manager = address(
            new ERC1967Proxy(address(implementation), abi.encodeCall(EmbeddingBSM.initialize, (token, registry)))
        );
        EmbeddingBSM(payable(manager)).configureModel(
            vm.parseJsonString(config, ".embedding.model"),
            uint32(vm.parseJsonUint(config, ".embedding.dimensions")),
            uint32(vm.parseJsonUint(config, ".embedding.max_sequence_length")),
            uint64(vm.parseJsonUint(config, ".embedding.price_per_1k_tokens")),
            uint32(vm.parseJsonUint(config, ".gpu.min_vram_mib"))
        );
        vm.stopBroadcast();
        console2.log("DEPLOY_MANAGER=%s", vm.toString(manager));
    }
}
