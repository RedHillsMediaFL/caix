import Foundation

/// Non-negotiable resident-memory admission plan for native Qwen3.8-27B on a 64 GiB M-series
/// machine. Unlike the Gemma planner this has no context ladder: a Qwen3.8 bundle is either
/// admitted with its complete 262,144-token state or denied before Core AI specializes it.
public struct Qwen38ResidentMemoryPlan: Sendable, Equatable {
    public enum Denial: Sendable, Equatable {
        case insufficientPhysicalMemory
        case insufficientAvailableMemory
        case memoryPressureNotGreen
        case swapGrowthExceeded
    }

    public enum Admission: Sendable, Equatable {
        case admit
        case deny(Denial)
    }

    public static let m1Ultra64GiB = Qwen38ResidentMemoryPlan(
        physicalMemoryBytes: 64 * Self.gib,
        targetWeightBudgetBytes: 20 * Self.gib + (2 * Self.gib) / 5,
        fixedHybridStateBudgetBytes: 160 * Self.mib,
        coreAIAndMTPBudgetBytes: 6 * Self.gib,
        headroomBytes: 8 * Self.gib,
        maximumSwapGrowthBytes: 256 * Self.mib)

    public let layout: Qwen38StateLayout
    public let physicalMemoryBytes: UInt64
    /// Q4 affine body + Q8 embeddings/head/tail MLP; intentionally above the 20 GiB source
    /// derivative to include Core AI packaged-weight variance.
    public let targetWeightBudgetBytes: UInt64
    /// Fixed FP16 convolution plus FP32 recurrent state for the 48 linear-attention layers.
    public let fixedHybridStateBudgetBytes: UInt64
    /// Native Qwen MTP sidecar, Core AI specialization, graph scratch, and GPU sampler buffers.
    public let coreAIAndMTPBudgetBytes: UInt64
    /// Memory left outside the worker so macOS never has to trade full-context state for swap.
    public let headroomBytes: UInt64
    public let maximumSwapGrowthBytes: UInt64

    public init(
        layout: Qwen38StateLayout = .native,
        physicalMemoryBytes: UInt64,
        targetWeightBudgetBytes: UInt64,
        fixedHybridStateBudgetBytes: UInt64,
        coreAIAndMTPBudgetBytes: UInt64,
        headroomBytes: UInt64,
        maximumSwapGrowthBytes: UInt64
    ) {
        self.layout = layout
        self.physicalMemoryBytes = physicalMemoryBytes
        self.targetWeightBudgetBytes = targetWeightBudgetBytes
        self.fixedHybridStateBudgetBytes = fixedHybridStateBudgetBytes
        self.coreAIAndMTPBudgetBytes = coreAIAndMTPBudgetBytes
        self.headroomBytes = headroomBytes
        self.maximumSwapGrowthBytes = maximumSwapGrowthBytes
    }

    public var contextLength: Int { layout.maxContextLength }
    /// Exposed to make it impossible for a generic caller to infer a smaller fallback tier.
    public var contextCandidates: [Int] { [contextLength] }
    public var keyValueCacheBytes: UInt64 { layout.keyValueCacheBytes }

    /// Expected process residency once all model assets and persistent states are loaded.
    public var plannedResidentBytes: UInt64 {
        targetWeightBudgetBytes
            + fixedHybridStateBudgetBytes
            + coreAIAndMTPBudgetBytes
            + keyValueCacheBytes
    }

    /// Available memory required before loading. This includes planned residency plus the OS
    /// headroom rather than assuming Core AI can safely evict or downsize the 262K state.
    public var requiredAvailableBytes: UInt64 { plannedResidentBytes + headroomBytes }

    /// Pure admission check; no Core AI model, NDArray, or MTLBuffer is allocated here.
    public func admit(
        availableBytes: UInt64,
        pressure: ResidentMemoryPressure,
        swapGrowthBytes: UInt64
    ) throws -> Admission {
        guard physicalMemoryBytes >= requiredAvailableBytes else {
            return .deny(.insufficientPhysicalMemory)
        }
        guard pressure == .green else { return .deny(.memoryPressureNotGreen) }
        guard swapGrowthBytes <= maximumSwapGrowthBytes else {
            return .deny(.swapGrowthExceeded)
        }
        guard availableBytes >= requiredAvailableBytes else {
            return .deny(.insufficientAvailableMemory)
        }
        return .admit
    }

    private static let mib: UInt64 = 1_048_576
    private static let gib: UInt64 = 1_073_741_824
}
