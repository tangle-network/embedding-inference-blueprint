// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import {Script, console2} from "forge-std/Script.sol";
import {EmbeddingBSM} from "../src/EmbeddingBSM.sol";
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";

/// Model-specific manager deployment; cargo-tangle registers the definition.
contract Deploy is Script {
    function run() external {
        address registry = vm.envAddress("TANGLE_CORE");
        require(registry.code.length != 0, "TANGLE_CORE has no code");
        address token = vm.envAddress("TSUSD_ADDRESS");
        vm.startBroadcast();
        EmbeddingBSM implementation = new EmbeddingBSM();
        address manager = address(
            new ERC1967Proxy(address(implementation), abi.encodeCall(EmbeddingBSM.initialize, (token, registry)))
        );
        console2.log("DEPLOY_EMBEDDING_BSM_IMPL=%s", vm.toString(address(implementation)));
        vm.stopBroadcast();
        console2.log("DEPLOY_EMBEDDING_BSM_PROXY=%s", vm.toString(manager));
    }
}
