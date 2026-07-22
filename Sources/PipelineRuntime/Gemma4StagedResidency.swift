import Foundation

/// Asset lifecycle promised by the locked Gemma 4 staged producer.
public enum DistributedAssetResidencyPolicy: String, Codable, Hashable, Sendable {
    case decodeSetPlusOneStreamedPrefillStage =
        "decode_set_plus_one_streamed_prefill_stage"
}

/// Context-tier selection promised by the locked Gemma 4 staged producer.
public enum DistributedRuntimeAllocationPolicy: String, Codable, Hashable, Sendable {
    case preflightInitialThenFallback = "preflight_initial_then_fallback"
}

/// Memory metadata carried by `stage-manifest.json` for the target-only resident runtime.
public struct DistributedRuntimeMemoryContract: Codable, Hashable, Sendable {
    public struct Tier: Codable, Hashable, Sendable {
        public let contextTokens: Int
        public let cacheBytes: UInt64
        public let minimumPhysicalMemoryBytes: UInt64

        enum CodingKeys: String, CodingKey {
            case contextTokens = "context_tokens"
            case cacheBytes = "cache_bytes"
            case minimumPhysicalMemoryBytes = "minimum_physical_memory_bytes"
        }
    }

    public let allocationPolicy: DistributedRuntimeAllocationPolicy
    public let assetResidencyPolicy: DistributedAssetResidencyPolicy
    public let fullPrefillAndDecodeSetsMayNotBeDualResident: Bool
    public let residentWeightUpperBoundBytes: UInt64
    public let runtimeOverheadReserveBytes: UInt64
    public let coresidentServicesReserveBytes: UInt64
    public let systemHeadroomBytes: UInt64
    public let initial: Tier
    public let fallback: Tier

    enum CodingKeys: String, CodingKey {
        case allocationPolicy = "allocation_policy"
        case assetResidencyPolicy = "asset_residency_policy"
        case fullPrefillAndDecodeSetsMayNotBeDualResident =
            "full_prefill_and_decode_sets_may_not_be_dual_resident"
        case residentWeightUpperBoundBytes = "resident_weight_upper_bound_bytes"
        case runtimeOverheadReserveBytes = "runtime_overhead_reserve_bytes"
        case coresidentServicesReserveBytes = "coresident_services_reserve_bytes"
        case systemHeadroomBytes = "system_headroom_bytes"
        case initial
        case fallback
    }

    func validate(stages: [DistributedStageManifestStage]) throws {
        guard fullPrefillAndDecodeSetsMayNotBeDualResident else {
            throw DistributedStageManifestError.invalidManifest(
                "runtime_memory.full_prefill_and_decode_sets_may_not_be_dual_resident must be true")
        }
        guard initial.contextTokens > fallback.contextTokens,
              fallback.contextTokens > 0
        else {
            throw DistributedStageManifestError.invalidManifest(
                "runtime_memory fallback context must be positive and smaller than initial")
        }
        guard initial.cacheBytes > 0, fallback.cacheBytes > 0,
              initial.cacheBytes >= fallback.cacheBytes,
              initial.minimumPhysicalMemoryBytes >= fallback.minimumPhysicalMemoryBytes,
              fallback.minimumPhysicalMemoryBytes > 0
        else {
            throw DistributedStageManifestError.invalidManifest(
                "runtime_memory fallback byte bounds must be positive and no larger than initial")
        }
        guard residentWeightUpperBoundBytes > 0,
              runtimeOverheadReserveBytes > 0,
              coresidentServicesReserveBytes > 0,
              systemHeadroomBytes > 0
        else {
            throw DistributedStageManifestError.invalidManifest(
                "runtime_memory resident reserve byte counts must be positive")
        }
        for stage in stages {
            guard let decodeAssetName = stage.decodeAssetName,
                  !decodeAssetName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  decodeAssetName != stage.assetName
            else {
                throw DistributedStageManifestError.invalidManifest(
                    "streamed prefill stage \(stage.id) requires a distinct decode_asset")
            }
        }
    }
}

public extension DistributedStageManifest {
    var requiresStreamedPrefillResidency: Bool {
        runtimeMemory?.assetResidencyPolicy == .decodeSetPlusOneStreamedPrefillStage
    }
}
