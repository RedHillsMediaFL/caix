import Foundation

/// Memory pressure signal used by pre-allocation and live service gates.
public enum ResidentMemoryPressure: String, Codable, Sendable, Equatable {
    case green
    case yellow
    case red
}

/// Analytic, allocation-free memory planner for the pinned Gemma 4 31B Q4_0 target/assistant
/// and Whisper large-v2 FP16 deployment.
///
/// The estimates are deliberately conservative budgets, not claims about measured RSS. Hardware
/// soak evidence may lower the context ceiling; it must never raise it without changing the
/// explicit budgets and tests here.
public struct ResidentMemoryPlanner: Sendable {
    public struct Budget: Sendable, Equatable {
        public var physicalMemoryBytes: UInt64
        public var plannedWorkerLimitBytes: UInt64
        public var requiredAvailableBytes: UInt64
        public var maximumSwapGrowthBytes: UInt64
        public var targetWeightBudgetBytes: UInt64
        public var assistantWeightBudgetBytes: UInt64
        public var whisperResidentBudgetBytes: UInt64
        public var coreAIAndServiceReserveBytes: UInt64

        public init(
            physicalMemoryBytes: UInt64,
            plannedWorkerLimitBytes: UInt64,
            requiredAvailableBytes: UInt64,
            maximumSwapGrowthBytes: UInt64,
            targetWeightBudgetBytes: UInt64,
            assistantWeightBudgetBytes: UInt64,
            whisperResidentBudgetBytes: UInt64,
            coreAIAndServiceReserveBytes: UInt64
        ) {
            self.physicalMemoryBytes = physicalMemoryBytes
            self.plannedWorkerLimitBytes = plannedWorkerLimitBytes
            self.requiredAvailableBytes = requiredAvailableBytes
            self.maximumSwapGrowthBytes = maximumSwapGrowthBytes
            self.targetWeightBudgetBytes = targetWeightBudgetBytes
            self.assistantWeightBudgetBytes = assistantWeightBudgetBytes
            self.whisperResidentBudgetBytes = whisperResidentBudgetBytes
            self.coreAIAndServiceReserveBytes = coreAIAndServiceReserveBytes
        }

        /// Conservative launch budget for this 64 GiB M1 Ultra.
        ///
        /// - 18 GiB target budget covers the 16.44 GiB official Q4_0 artifact plus packaging.
        /// - 1.25 GiB assistant budget includes a duplicated Q4 target embedding for fast drafts.
        /// - 4.5 GiB Whisper budget covers FP16 weights, persistent KV, and runtime overhead.
        /// - 6 GiB remains for Core AI specialization, stage transients, logits, and services.
        public static let studio64GiB = Budget(
            physicalMemoryBytes: 64 * Self.gib,
            plannedWorkerLimitBytes: 40 * Self.gib,
            requiredAvailableBytes: 8 * Self.gib,
            maximumSwapGrowthBytes: 256 * Self.mib,
            targetWeightBudgetBytes: 18 * Self.gib,
            assistantWeightBudgetBytes: 5 * Self.gib / 4,
            whisperResidentBudgetBytes: 9 * Self.gib / 2,
            coreAIAndServiceReserveBytes: 6 * Self.gib)

        private static let mib: UInt64 = 1_048_576
        private static let gib: UInt64 = 1_073_741_824
    }

    public struct Estimate: Sendable, Equatable {
        public let contextLength: Int
        public let fixedResidentBytes: UInt64
        public let targetGlobalKVBytes: UInt64
        public let targetSlidingKVBytes: UInt64
        public let representativeGlobalKVBytes: UInt64
        public let representativeSlidingKVBytes: UInt64
        public let plannedResidentBytes: UInt64
    }

    public enum CandidateRejection: String, Sendable, Equatable {
        case plannedWorkerLimit
        case physicalMemoryHeadroom
    }

    public enum PreconditionFailure: String, Sendable, Equatable {
        case insufficientAvailableMemory
        case memoryPressureNotGreen
        case swapGrowthExceeded
    }

    public struct CandidateEvaluation: Sendable, Equatable {
        public let contextLength: Int
        public let estimate: Estimate
        public let rejection: CandidateRejection?

        /// This planner performs arithmetic only. Keeping the fact explicit prevents a future
        /// caller from confusing candidate evaluation with Core AI state allocation.
        public let allocated = false
    }

    public struct Selection: Sendable, Equatable {
        public let selectedContextLength: Int?
        public let evaluations: [CandidateEvaluation]
        public let preconditionFailure: PreconditionFailure?
    }

    public enum Error: Swift.Error, Sendable, Equatable {
        case invalidContextLength
        case arithmeticOverflow
        case invalidContextCandidates
    }

    public let budget: Budget
    public let contextCandidates: [Int]

    public init(
        budget: Budget,
        contextCandidates: [Int] = [262_144, 131_072, 65_536, 32_768, 16_384]
    ) {
        self.budget = budget
        self.contextCandidates = contextCandidates
    }

    public static let studio64GiB = ResidentMemoryPlanner(budget: .studio64GiB)

    /// Estimates split target KV and the two representative MTP caches without allocating them.
    public func estimate(contextLength: Int) throws -> Estimate {
        guard contextLength > 0 else { throw Error.invalidContextLength }
        let context = UInt64(contextLength)

        // Ten global layers: K+V, four KV heads, 512 dimensions, FP16.
        let targetGlobal = try Self.product([10, 2, 4, 512, 2, context])
        // Fifty local layers retain the 1,024-token window plus one 512-token prefill chunk.
        let targetSliding = try Self.product([50, 2, 16, 256, 2, 1_536])
        // EAGLE representative K/V from one full layer and one bounded sliding layer.
        let representativeGlobal = try Self.product([2, 4, 512, 2, context])
        let representativeSliding = try Self.product([2, 16, 256, 2, 1_024])
        let fixed = try Self.sum([
            budget.targetWeightBudgetBytes,
            budget.assistantWeightBudgetBytes,
            budget.whisperResidentBudgetBytes,
            budget.coreAIAndServiceReserveBytes,
        ])
        let planned = try Self.sum([
            fixed,
            targetGlobal,
            targetSliding,
            representativeGlobal,
            representativeSliding,
        ])
        return Estimate(
            contextLength: contextLength,
            fixedResidentBytes: fixed,
            targetGlobalKVBytes: targetGlobal,
            targetSlidingKVBytes: targetSliding,
            representativeGlobalKVBytes: representativeGlobal,
            representativeSlidingKVBytes: representativeSliding,
            plannedResidentBytes: planned)
    }

    /// Walks the locked ladder from largest to smallest and stops at the first comfortable tier.
    /// Rejected larger tiers are evidence only; no Core AI object or backing buffer is created.
    public func selectLargestComfortable(
        availableBytes: UInt64,
        pressure: ResidentMemoryPressure,
        swapGrowthBytes: UInt64
    ) throws -> Selection {
        guard availableBytes >= budget.requiredAvailableBytes else {
            return Selection(
                selectedContextLength: nil,
                evaluations: [],
                preconditionFailure: .insufficientAvailableMemory)
        }
        guard pressure == .green else {
            return Selection(
                selectedContextLength: nil,
                evaluations: [],
                preconditionFailure: .memoryPressureNotGreen)
        }
        guard swapGrowthBytes <= budget.maximumSwapGrowthBytes else {
            return Selection(
                selectedContextLength: nil,
                evaluations: [],
                preconditionFailure: .swapGrowthExceeded)
        }
        guard Self.isStrictDescendingUniquePositive(contextCandidates) else {
            throw Error.invalidContextCandidates
        }

        var evaluations: [CandidateEvaluation] = []
        for contextLength in contextCandidates {
            let estimate = try estimate(contextLength: contextLength)
            let rejection: CandidateRejection?
            if estimate.plannedResidentBytes > budget.plannedWorkerLimitBytes {
                rejection = .plannedWorkerLimit
            } else {
                let withHeadroom = try Self.sum([
                    estimate.plannedResidentBytes,
                    budget.requiredAvailableBytes,
                ])
                rejection = withHeadroom > budget.physicalMemoryBytes
                    ? .physicalMemoryHeadroom : nil
            }
            evaluations.append(CandidateEvaluation(
                contextLength: contextLength,
                estimate: estimate,
                rejection: rejection))
            if rejection == nil {
                return Selection(
                    selectedContextLength: contextLength,
                    evaluations: evaluations,
                    preconditionFailure: nil)
            }
        }
        return Selection(
            selectedContextLength: nil,
            evaluations: evaluations,
            preconditionFailure: nil)
    }

    private static func product(_ factors: [UInt64]) throws -> UInt64 {
        try factors.reduce(1) { partial, value in
            let result = partial.multipliedReportingOverflow(by: value)
            guard !result.overflow else { throw Error.arithmeticOverflow }
            return result.partialValue
        }
    }

    private static func sum(_ values: [UInt64]) throws -> UInt64 {
        try values.reduce(0) { partial, value in
            let result = partial.addingReportingOverflow(value)
            guard !result.overflow else { throw Error.arithmeticOverflow }
            return result.partialValue
        }
    }

    private static func isStrictDescendingUniquePositive(_ values: [Int]) -> Bool {
        guard let first = values.first, first > 0 else { return false }
        return zip(values, values.dropFirst()).allSatisfy { current, next in
            next > 0 && current > next
        }
    }
}

/// Live admission gate for the separately supervised resident workers.
public struct ResidentServiceHealthGate: Sendable {
    public struct Limits: Sendable, Equatable {
        public let drainResidentBytes: UInt64
        public let restartResidentBytes: UInt64
        public let requiredAvailableBytes: UInt64
        public let maximumSwapGrowthBytes: UInt64

        public init(
            drainResidentBytes: UInt64,
            restartResidentBytes: UInt64,
            requiredAvailableBytes: UInt64,
            maximumSwapGrowthBytes: UInt64
        ) {
            self.drainResidentBytes = drainResidentBytes
            self.restartResidentBytes = restartResidentBytes
            self.requiredAvailableBytes = requiredAvailableBytes
            self.maximumSwapGrowthBytes = maximumSwapGrowthBytes
        }

        public static let studio64GiB = Limits(
            drainResidentBytes: 42 * 1_073_741_824,
            restartResidentBytes: 44 * 1_073_741_824,
            requiredAvailableBytes: 8 * 1_073_741_824,
            maximumSwapGrowthBytes: 256 * 1_048_576)
    }

    public struct Snapshot: Sendable, Equatable {
        public let workerResidentBytes: UInt64
        public let availableBytes: UInt64
        public let pressure: ResidentMemoryPressure
        /// Growth since this supervisor generation began, not lifetime system swap usage.
        public let swapGrowthBytes: UInt64

        public init(
            workerResidentBytes: UInt64,
            availableBytes: UInt64,
            pressure: ResidentMemoryPressure,
            swapGrowthBytes: UInt64
        ) {
            self.workerResidentBytes = workerResidentBytes
            self.availableBytes = availableBytes
            self.pressure = pressure
            self.swapGrowthBytes = swapGrowthBytes
        }
    }

    public enum DrainReason: String, Sendable, Equatable {
        case residentLimit
        case insufficientAvailableMemory
        case memoryPressure
        case swapGrowth
    }

    public enum RestartReason: String, Sendable, Equatable {
        case residentKillLimit
    }

    public enum Action: Sendable, Equatable {
        case admit
        case drain(DrainReason)
        case restart(RestartReason)
    }

    public let limits: Limits

    public init(limits: Limits) {
        self.limits = limits
    }

    public func action(for snapshot: Snapshot) -> Action {
        if snapshot.workerResidentBytes >= limits.restartResidentBytes {
            return .restart(.residentKillLimit)
        }
        if snapshot.workerResidentBytes >= limits.drainResidentBytes {
            return .drain(.residentLimit)
        }
        if snapshot.pressure != .green {
            return .drain(.memoryPressure)
        }
        if snapshot.availableBytes < limits.requiredAvailableBytes {
            return .drain(.insufficientAvailableMemory)
        }
        if snapshot.swapGrowthBytes > limits.maximumSwapGrowthBytes {
            return .drain(.swapGrowth)
        }
        return .admit
    }
}
