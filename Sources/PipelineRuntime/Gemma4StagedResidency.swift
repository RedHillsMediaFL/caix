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

/// Allocation-free live inputs used before selecting a context tier or loading an asset.
public struct DistributedStagedMemorySnapshot: Sendable, Equatable {
    public let totalPhysicalMemoryBytes: UInt64
    public let workerResidentBytes: UInt64
    public let availableBytes: UInt64
    public let pressure: ResidentMemoryPressure
    public let swapGrowthBytes: UInt64

    public init(
        totalPhysicalMemoryBytes: UInt64,
        workerResidentBytes: UInt64,
        availableBytes: UInt64,
        pressure: ResidentMemoryPressure,
        swapGrowthBytes: UInt64
    ) {
        self.totalPhysicalMemoryBytes = totalPhysicalMemoryBytes
        self.workerResidentBytes = workerResidentBytes
        self.availableBytes = availableBytes
        self.pressure = pressure
        self.swapGrowthBytes = swapGrowthBytes
    }
}

public enum DistributedStagedContextTier: String, Sendable, Equatable {
    case initial
    case fallback
}

public struct DistributedStagedContextSelection: Sendable, Equatable {
    public let tier: DistributedStagedContextTier
    public let contextTokens: Int

    public init(tier: DistributedStagedContextTier, contextTokens: Int) {
        self.tier = tier
        self.contextTokens = contextTokens
    }
}

public enum DistributedStagedMemoryAdmissionError: Error, Sendable, Equatable {
    case telemetryUnavailable
    case drain(ResidentServiceHealthGate.DrainReason)
    case restart(ResidentServiceHealthGate.RestartReason)
    case insufficientPhysicalMemory(requiredBytes: UInt64, actualBytes: UInt64)
    case insufficientAvailableMemory(requiredBytes: UInt64, actualBytes: UInt64)
}

/// Live, repeatable admission check for resident decode startup and every transient prefill load.
/// The provider performs telemetry only; rejected context tiers never allocate CoreAI state.
public struct DistributedStagedMemoryAdmission {
    private let contract: DistributedRuntimeMemoryContract
    private let gate: ResidentServiceHealthGate
    private let snapshotProvider: () throws -> DistributedStagedMemorySnapshot

    public init(
        contract: DistributedRuntimeMemoryContract,
        limits: ResidentServiceHealthGate.Limits = .studio64GiB,
        snapshotProvider: @escaping () throws -> DistributedStagedMemorySnapshot
    ) {
        self.contract = contract
        self.gate = ResidentServiceHealthGate(limits: limits)
        self.snapshotProvider = snapshotProvider
    }

    public func selectContext() throws -> DistributedStagedContextSelection {
        let snapshot = try checkedSnapshot()
        if snapshot.totalPhysicalMemoryBytes >= contract.initial.minimumPhysicalMemoryBytes,
           snapshot.availableBytes >= availableRequirement(
               for: contract.initial,
               snapshot: snapshot)
        {
            return DistributedStagedContextSelection(
                tier: .initial,
                contextTokens: contract.initial.contextTokens)
        }
        if snapshot.totalPhysicalMemoryBytes >= contract.fallback.minimumPhysicalMemoryBytes,
           snapshot.availableBytes >= availableRequirement(
               for: contract.fallback,
               snapshot: snapshot)
        {
            return DistributedStagedContextSelection(
                tier: .fallback,
                contextTokens: contract.fallback.contextTokens)
        }
        if snapshot.totalPhysicalMemoryBytes < contract.fallback.minimumPhysicalMemoryBytes {
            throw DistributedStagedMemoryAdmissionError.insufficientPhysicalMemory(
                requiredBytes: contract.fallback.minimumPhysicalMemoryBytes,
                actualBytes: snapshot.totalPhysicalMemoryBytes)
        }
        throw DistributedStagedMemoryAdmissionError.insufficientAvailableMemory(
            requiredBytes: availableRequirement(for: contract.fallback, snapshot: snapshot),
            actualBytes: snapshot.availableBytes)
    }

    public func checkBeforeAssetLoad() throws {
        _ = try checkedSnapshot()
    }

    private func checkedSnapshot() throws -> DistributedStagedMemorySnapshot {
        let snapshot = try snapshotProvider()
        guard snapshot.totalPhysicalMemoryBytes > 0,
              snapshot.workerResidentBytes > 0,
              snapshot.availableBytes > 0
        else {
            throw DistributedStagedMemoryAdmissionError.telemetryUnavailable
        }
        switch gate.action(for: ResidentServiceHealthGate.Snapshot(
            workerResidentBytes: snapshot.workerResidentBytes,
            availableBytes: snapshot.availableBytes,
            pressure: snapshot.pressure,
            swapGrowthBytes: snapshot.swapGrowthBytes))
        {
        case .admit:
            return snapshot
        case .drain(let reason):
            throw DistributedStagedMemoryAdmissionError.drain(reason)
        case .restart(let reason):
            throw DistributedStagedMemoryAdmissionError.restart(reason)
        }
    }

    /// Projects only the additional worker footprint not already present in the live RSS.
    /// `availableBytes` already excludes the live OS and co-resident services; adding their
    /// reserves again would double-count them. The tier's minimum physical-memory floor retains
    /// the whole-machine reserve contract, while `checkBeforeAssetLoad()` preserves the live
    /// green-pressure, 8 GiB-available, and swap-growth gates before every specialization.
    private func availableRequirement(
        for tier: DistributedRuntimeMemoryContract.Tier,
        snapshot: DistributedStagedMemorySnapshot
    ) -> UInt64 {
        let projectedWorkerBytes = Self.saturatedSum([
            contract.residentWeightUpperBoundBytes,
            contract.runtimeOverheadReserveBytes,
            tier.cacheBytes,
        ])
        let incrementalWorkerBytes = projectedWorkerBytes > snapshot.workerResidentBytes
            ? projectedWorkerBytes - snapshot.workerResidentBytes
            : 0
        return incrementalWorkerBytes
    }

    private static func saturatedSum(_ values: [UInt64]) -> UInt64 {
        values.reduce(0) { partial, value in
            let (sum, overflow) = partial.addingReportingOverflow(value)
            return overflow ? UInt64.max : sum
        }
    }
}

/// A stage whose one-token decode model and request KV state stay resident while its dynamic
/// prefill model is held only between `loadPrefill` and `unloadPrefill`.
public protocol DistributedDecodeResidentStageHandle: DistributedStageHandle {
    var isPrefillResident: Bool { get }
    func loadPrefill() async throws
    func unloadPrefill()
}
