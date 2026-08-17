/// Honest execution routing for speculative backends.
///
/// Greedy verification can preserve target-only argmax output exactly. Sampled speculative
/// decoding additionally requires draft probabilities, probabilistic acceptance, and residual
/// correction sampling. Until that probability-bearing path is wired end to end, sampled
/// requests must use the target alone rather than silently changing the requested distribution.
public enum SpeculativeExecutionPolicy: String, Sendable, Equatable {
    case mtpGreedy
    case targetOnlySampled

    public static func route(temperature: Double) -> SpeculativeExecutionPolicy {
        temperature > 0 ? .targetOnlySampled : .mtpGreedy
    }

    public static func route(options: CoreAIPipeline.Options) -> SpeculativeExecutionPolicy {
        route(temperature: options.temperature)
    }
}
