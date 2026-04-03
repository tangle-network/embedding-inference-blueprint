// SPDX-License-Identifier: MIT
pragma solidity ^0.8.26;

import { EnumerableSet } from "@openzeppelin/contracts/utils/structs/EnumerableSet.sol";
import { Initializable } from "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import { UUPSUpgradeable } from "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";
import { BlueprintServiceManagerBase } from "tnt-core/BlueprintServiceManagerBase.sol";

/// @title EmbeddingBSM
/// @notice Blueprint Service Manager for embedding inference services.
/// @dev Operators register with GPU capabilities and embedding model metadata.
///      Pricing is per-1K tokens. Only accepts tsUSD (ShieldedCredits wrapped token).
contract EmbeddingBSM is Initializable, UUPSUpgradeable, BlueprintServiceManagerBase {
    using EnumerableSet for EnumerableSet.AddressSet;

    // ═══════════════════════════════════════════════════════════════════════
    // ERRORS
    // ═══════════════════════════════════════════════════════════════════════

    error InvalidPaymentAsset(address asset);
    error InsufficientGpuCapability(uint32 required, uint32 provided);
    error ModelNotSupported(string model);
    error InvalidEmbeddingDimensions(uint32 expected, uint32 provided);
    error SequenceLengthExceeded(uint32 maxAllowed, uint32 provided);
    error OperatorNotRegistered(address operator);

    // ═══════════════════════════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════════════════════════

    event OperatorRegistered(
        address indexed operator,
        string model,
        uint32 dimensions,
        uint32 maxSeqLen,
        uint32 gpuCount,
        uint32 totalVramMib
    );
    event ModelConfigured(
        string model,
        uint32 dimensions,
        uint32 maxSequenceLength,
        uint64 pricePer1kTokens
    );
    event EmbeddingJobSubmitted(uint64 indexed serviceId, uint64 indexed jobCallId, uint32 inputCount);
    event EmbeddingResultSubmitted(uint64 indexed serviceId, uint64 indexed jobCallId, uint32 totalTokens);

    // ═══════════════════════════════════════════════════════════════════════
    // TYPES
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Capabilities reported by operator at registration
    struct OperatorCapabilities {
        string model;
        uint32 embeddingDimensions;
        uint32 maxSequenceLength;
        uint32 gpuCount;
        uint32 totalVramMib;
        string gpuModel;
        string endpoint;
        bool active;
    }

    /// @notice Model pricing and metadata
    struct ModelConfig {
        uint32 dimensions;
        uint32 maxSequenceLength;
        uint64 pricePer1kTokens;     // in tsUSD base units (6 decimals)
        uint32 minGpuVramMib;
        bool enabled;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // STATE
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice The only accepted payment token (tsUSD)
    address public tsUSD;

    /// @notice Minimum operator stake (in TNT)
    uint256 public constant MIN_OPERATOR_STAKE = 50 ether;

    /// @notice operator => capabilities
    mapping(address => OperatorCapabilities) public operatorCaps;

    /// @notice model name hash => ModelConfig
    mapping(bytes32 => ModelConfig) public modelConfigs;

    /// @notice Set of registered operators
    EnumerableSet.AddressSet private _operators;

    // ═══════════════════════════════════════════════════════════════════════
    // INITIALIZATION
    // ═══════════════════════════════════════════════════════════════════════

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    /// @notice Initialize the contract (called once via proxy)
    /// @param _tsUSD The wrapped stablecoin accepted for payment
    function initialize(address _tsUSD) external initializer {
        __UUPSUpgradeable_init();
        tsUSD = _tsUSD;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ADMIN
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Configure an embedding model's pricing and requirements
    function configureModel(
        string calldata model,
        uint32 dimensions,
        uint32 maxSequenceLength,
        uint64 pricePer1kTokens,
        uint32 minGpuVramMib
    ) external onlyBlueprintOwner {
        bytes32 key = keccak256(bytes(model));
        modelConfigs[key] = ModelConfig({
            dimensions: dimensions,
            maxSequenceLength: maxSequenceLength,
            pricePer1kTokens: pricePer1kTokens,
            minGpuVramMib: minGpuVramMib,
            enabled: true
        });

        emit ModelConfigured(model, dimensions, maxSequenceLength, pricePer1kTokens);
    }

    /// @notice Disable a model
    function disableModel(string calldata model) external onlyBlueprintOwner {
        bytes32 key = keccak256(bytes(model));
        modelConfigs[key].enabled = false;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // OPERATOR LIFECYCLE
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Operator registers with embedding capabilities.
    /// @param registrationInputs abi.encode(string model, uint32 dimensions, uint32 maxSeqLen, uint32 gpuCount, uint32 totalVramMib, string gpuModel, string endpoint)
    function onRegister(address operator, bytes calldata registrationInputs) external payable override onlyFromTangle {
        (
            string memory model,
            uint32 dimensions,
            uint32 maxSeqLen,
            uint32 gpuCount,
            uint32 totalVramMib,
            string memory gpuModel,
            string memory endpoint
        ) = abi.decode(registrationInputs, (string, uint32, uint32, uint32, uint32, string, string));

        // Validate model is configured
        bytes32 modelKey = keccak256(bytes(model));
        ModelConfig storage mc = modelConfigs[modelKey];
        if (!mc.enabled) revert ModelNotSupported(model);

        // Validate embedding dimensions match model config
        if (dimensions != mc.dimensions) {
            revert InvalidEmbeddingDimensions(mc.dimensions, dimensions);
        }

        // Validate GPU capabilities meet model requirements
        if (totalVramMib < mc.minGpuVramMib) {
            revert InsufficientGpuCapability(mc.minGpuVramMib, totalVramMib);
        }

        operatorCaps[operator] = OperatorCapabilities({
            model: model,
            embeddingDimensions: dimensions,
            maxSequenceLength: maxSeqLen,
            gpuCount: gpuCount,
            totalVramMib: totalVramMib,
            gpuModel: gpuModel,
            endpoint: endpoint,
            active: true
        });

        _operators.add(operator);

        emit OperatorRegistered(operator, model, dimensions, maxSeqLen, gpuCount, totalVramMib);
    }

    function onUnregister(address operator) external override onlyFromTangle {
        operatorCaps[operator].active = false;
        _operators.remove(operator);
    }

    function onUpdatePreferences(address operator, bytes calldata newPreferences) external payable override onlyFromTangle {
        string memory newEndpoint = abi.decode(newPreferences, (string));
        operatorCaps[operator].endpoint = newEndpoint;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SERVICE LIFECYCLE
    // ═══════════════════════════════════════════════════════════════════════

    function onRequest(
        uint64,
        address,
        address[] calldata,
        bytes calldata,
        uint64,
        address paymentAsset,
        uint256
    ) external payable override onlyFromTangle {
        if (paymentAsset != tsUSD && paymentAsset != address(0)) {
            revert InvalidPaymentAsset(paymentAsset);
        }
    }

    function onServiceInitialized(
        uint64,
        uint64,
        uint64 serviceId,
        address,
        address[] calldata,
        uint64
    ) external override onlyFromTangle {
        _permitAsset(serviceId, tsUSD);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // DYNAMIC MEMBERSHIP
    // ═══════════════════════════════════════════════════════════════════════

    function canJoin(uint64, address operator) external view override returns (bool) {
        return operatorCaps[operator].active;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // JOB LIFECYCLE
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Validate an embedding job submission
    /// @dev inputs = abi.encode(string[] inputs)
    function onJobCall(
        uint64 serviceId,
        uint8,
        uint64 jobCallId,
        bytes calldata inputs
    ) external payable override onlyFromTangle {
        string[] memory inputTexts = abi.decode(inputs, (string[]));
        uint32 inputCount = uint32(inputTexts.length);

        emit EmbeddingJobSubmitted(serviceId, jobCallId, inputCount);
    }

    /// @notice Validate an embedding job result
    /// @dev outputs = abi.encode(uint32 count, uint32 totalTokens, uint32 dimensions)
    function onJobResult(
        uint64 serviceId,
        uint8,
        uint64 jobCallId,
        address operator,
        bytes calldata,
        bytes calldata outputs
    ) external payable override onlyFromTangle {
        if (!operatorCaps[operator].active) revert OperatorNotRegistered(operator);

        (, uint32 totalTokens,) = abi.decode(outputs, (uint32, uint32, uint32));

        emit EmbeddingResultSubmitted(serviceId, jobCallId, totalTokens);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // QUERIES
    // ═══════════════════════════════════════════════════════════════════════

    function queryIsPaymentAssetAllowed(uint64 serviceId, address asset) external view override returns (bool) {
        if (asset == address(0)) return true;
        address[] memory permitted = _getPermittedAssets(serviceId);
        if (permitted.length == 0) return asset == tsUSD;
        for (uint256 i; i < permitted.length; ++i) {
            if (permitted[i] == asset) return true;
        }
        return false;
    }

    function getAggregationThreshold(uint64, uint8) external pure override returns (uint16, uint8) {
        return (0, 0);
    }

    function getMinOperatorStake() external pure override returns (bool, uint256) {
        return (false, MIN_OPERATOR_STAKE);
    }

    function getHeartbeatInterval(uint64) external pure override returns (bool, uint64) {
        return (false, 100);
    }

    function getExitConfig(uint64) external pure override returns (bool, uint64, uint64, bool) {
        return (false, 3600, 3600, true);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // VIEW HELPERS
    // ═══════════════════════════════════════════════════════════════════════

    /// @notice Get all registered operators
    function getOperators() external view returns (address[] memory) {
        return _operators.values();
    }

    /// @notice Get operator count
    function getOperatorCount() external view returns (uint256) {
        return _operators.length();
    }

    /// @notice Get model config by name
    function getModelConfig(string calldata model) external view returns (ModelConfig memory) {
        return modelConfigs[keccak256(bytes(model))];
    }

    /// @notice Check if an operator is registered and active
    function isOperatorActive(address operator) external view returns (bool) {
        return operatorCaps[operator].active;
    }

    /// @notice Get operator pricing for a given operator address.
    function getOperatorPricing(address operator)
        external
        view
        returns (uint64 pricePer1kTokens, uint32 dimensions, string memory endpoint)
    {
        OperatorCapabilities storage caps = operatorCaps[operator];
        if (!caps.active) revert OperatorNotRegistered(operator);

        bytes32 modelKey = keccak256(bytes(caps.model));
        ModelConfig storage mc = modelConfigs[modelKey];

        return (mc.pricePer1kTokens, mc.dimensions, caps.endpoint);
    }

    /// @notice Get operator's embedding dimensions
    function getOperatorDimensions(address operator) external view returns (uint32) {
        OperatorCapabilities storage caps = operatorCaps[operator];
        if (!caps.active) revert OperatorNotRegistered(operator);
        return caps.embeddingDimensions;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // UPGRADES
    // ═══════════════════════════════════════════════════════════════════════

    function _authorizeUpgrade(address) internal override onlyBlueprintOwner {}

    receive() external payable override {}
}
