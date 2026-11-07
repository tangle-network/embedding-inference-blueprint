// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import {Test} from "forge-std/Test.sol";
import {Deploy} from "../script/Deploy.s.sol";
import {EmbeddingBSM} from "../src/EmbeddingBSM.sol";
import {BlueprintServiceManagerBase} from "tnt-core/BlueprintServiceManagerBase.sol";

contract EmbeddingBSMTest is Test {
    EmbeddingBSM public bsm;
    address public tsUSD;

    address public tangleCore = address(0xC0DE);
    address public owner;
    address public operator1 = address(0x1111);
    address public operator2 = address(0x2222);
    address public user = address(0x3333);

    function setUp() public {
        tsUSD = address(0xAAAA);
        vm.etch(tangleCore, hex"00");
        vm.setEnv("TANGLE_CORE", vm.toString(tangleCore));
        vm.setEnv("TSUSD_ADDRESS", vm.toString(tsUSD));
        // The shipped deploy script, configured from deploy/config.example.json.
        bsm = EmbeddingBSM(payable(new Deploy().run()));
        owner = bsm.blueprintOwner();
        vm.prank(tangleCore);
        bsm.onBlueprintCreated(1, owner, tangleCore);
    }

    // --- Initialization ---

    function test_initialization() public view {
        assertEq(bsm.blueprintId(), 1);
        assertEq(bsm.blueprintOwner(), owner);
        assertEq(bsm.tangleCore(), tangleCore);
        assertEq(bsm.tsUSD(), tsUSD);
    }

    function test_cannotReinitialize() public {
        vm.expectRevert(BlueprintServiceManagerBase.AlreadyInitialized.selector);
        vm.prank(tangleCore);
        bsm.onBlueprintCreated(2, owner, tangleCore);
    }

    // --- Model Configuration ---

    function test_configureModel() public view {
        EmbeddingBSM.ModelConfig memory mc = bsm.getModelConfig("BAAI/bge-large-en-v1.5");
        assertEq(mc.dimensions, 1024);
        assertEq(mc.maxSequenceLength, 512);
        assertEq(mc.pricePer1kTokens, 10);
        assertEq(mc.minGpuVramMib, 4_096);
        assertTrue(mc.enabled);
    }

    function test_configureModel_onlyOwner() public {
        vm.prank(user);
        vm.expectRevert(
            abi.encodeWithSelector(BlueprintServiceManagerBase.OnlyBlueprintOwnerAllowed.selector, user, owner)
        );
        bsm.configureModel("test", 768, 256, 5, 4000);
    }

    function test_disableModel() public {
        vm.prank(owner);
        bsm.disableModel("BAAI/bge-large-en-v1.5");

        EmbeddingBSM.ModelConfig memory mc = bsm.getModelConfig("BAAI/bge-large-en-v1.5");
        assertFalse(mc.enabled);
    }

    // --- Operator Registration ---

    function test_registerOperator() public {
        _registerOperator(operator1);

        assertTrue(bsm.isOperatorActive(operator1));
        assertEq(bsm.getOperatorCount(), 1);
        assertEq(bsm.getOperatorDimensions(operator1), 1024);
    }

    function test_registerOperator_dimensionsMismatch() public {
        bytes memory regData = abi.encode(
            "BAAI/bge-large-en-v1.5",
            uint32(768), // wrong dimensions (should be 1024)
            uint32(512),
            uint32(2),
            uint32(16_000),
            "NVIDIA A100",
            "https://op1.example.com"
        );

        vm.prank(tangleCore);
        vm.expectRevert(abi.encodeWithSelector(EmbeddingBSM.InvalidEmbeddingDimensions.selector, 1024, 768));
        bsm.onRegister(operator1, regData);
    }

    function test_registerOperator_unsupportedModel() public {
        bytes memory regData = abi.encode(
            "nonexistent-model",
            uint32(768),
            uint32(512),
            uint32(1),
            uint32(8_000),
            "NVIDIA A100",
            "https://op1.example.com"
        );

        vm.prank(tangleCore);
        vm.expectRevert(abi.encodeWithSelector(EmbeddingBSM.ModelNotSupported.selector, "nonexistent-model"));
        bsm.onRegister(operator1, regData);
    }

    function test_registerOperator_insufficientVram() public {
        bytes memory regData = abi.encode(
            "BAAI/bge-large-en-v1.5",
            uint32(1024),
            uint32(512),
            uint32(1),
            uint32(2_048), // below the 4_096 minimum
            "NVIDIA RTX 3060",
            "https://op1.example.com"
        );

        vm.prank(tangleCore);
        vm.expectRevert(abi.encodeWithSelector(EmbeddingBSM.InsufficientGpuCapability.selector, 4_096, 2_048));
        bsm.onRegister(operator1, regData);
    }

    function test_registerOperator_onlyTangle() public {
        bytes memory regData = abi.encode(
            "BAAI/bge-large-en-v1.5",
            uint32(1024),
            uint32(512),
            uint32(1),
            uint32(8_000),
            "NVIDIA A100",
            "https://op1.example.com"
        );

        vm.prank(user);
        vm.expectRevert(
            abi.encodeWithSelector(BlueprintServiceManagerBase.OnlyTangleAllowed.selector, user, tangleCore)
        );
        bsm.onRegister(operator1, regData);
    }

    // --- Unregistration ---

    function test_unregisterOperator() public {
        _registerOperator(operator1);

        vm.prank(tangleCore);
        bsm.onUnregister(operator1);

        assertFalse(bsm.isOperatorActive(operator1));
        assertEq(bsm.getOperatorCount(), 0);
    }

    // --- Pricing ---

    function test_getOperatorPricing() public {
        _registerOperator(operator1);

        (uint64 price, uint32 dims, string memory endpoint) = bsm.getOperatorPricing(operator1);
        assertEq(price, 10);
        assertEq(dims, 1024);
        assertEq(keccak256(bytes(endpoint)), keccak256(bytes("https://op1.example.com")));
    }

    function test_getOperatorPricing_unregistered() public {
        vm.expectRevert(abi.encodeWithSelector(EmbeddingBSM.OperatorNotRegistered.selector, operator1));
        bsm.getOperatorPricing(operator1);
    }

    // --- Configuration Queries ---

    function test_minOperatorStake() public view {
        (bool useDefault, uint256 minStake) = bsm.getMinOperatorStake();
        assertFalse(useDefault);
        assertEq(minStake, 50 ether);
    }

    function test_canJoin() public {
        _registerOperator(operator1);
        assertTrue(bsm.canJoin(1, operator1));
        assertFalse(bsm.canJoin(1, operator2));
    }

    function _registerOperator(address op) internal {
        bytes memory regData = abi.encode(
            "BAAI/bge-large-en-v1.5",
            uint32(1024),
            uint32(512),
            uint32(2),
            uint32(16_000),
            "NVIDIA A100",
            "https://op1.example.com"
        );
        vm.prank(tangleCore);
        bsm.onRegister(op, regData);
    }
}
