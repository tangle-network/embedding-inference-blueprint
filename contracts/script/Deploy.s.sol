// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;
import {RegisterBlueprint} from "tangle-inference-core/script/RegisterBlueprint.s.sol";
import {console2} from "forge-std/Script.sol";
import {EmbeddingBSM} from "../src/EmbeddingBSM.sol";
import {ERC1967Proxy} from "@openzeppelin/contracts/proxy/ERC1967/ERC1967Proxy.sol";

contract Deploy is RegisterBlueprint {
    function _deployManager() internal override returns (address) {
        EmbeddingBSM impl = new EmbeddingBSM();
        console2.log("DEPLOY_EMBEDDING_BSM_IMPL=%s", vm.toString(address(impl)));
        return address(
            new ERC1967Proxy(address(impl), abi.encodeCall(EmbeddingBSM.initialize, (vm.envAddress("TSUSD_ADDRESS"))))
        );
    }

    function _report(address manager, uint64 blueprintId) internal override {
        console2.log("DEPLOY_EMBEDDING_BSM_PROXY=%s", vm.toString(manager));
        console2.log("DEPLOY_EMBEDDING_BLUEPRINT_ID=%s", vm.toString(blueprintId));
    }
}
