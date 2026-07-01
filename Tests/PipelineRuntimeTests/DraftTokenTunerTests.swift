import Testing
@testable import PipelineRuntime

@Test func draftTokenTunerClimbsAfterSustainedFullAcceptance() {
    var tuner = DraftTokenTuner(initial: 2, maximum: 5)

    #expect(tuner.current == 2)
    tuner.observe(accepted: 2, drafted: 2)
    #expect(tuner.current == 2)
    tuner.observe(accepted: 2, drafted: 2)
    #expect(tuner.current == 3)
    tuner.observe(accepted: 3, drafted: 3)
    tuner.observe(accepted: 3, drafted: 3)
    #expect(tuner.current == 4)
}

@Test func draftTokenTunerDropsOnPoorAcceptance() {
    var tuner = DraftTokenTuner(initial: 4, maximum: 6)

    tuner.observe(accepted: 1, drafted: 4)
    #expect(tuner.current == 3)
    tuner.observe(accepted: 1, drafted: 3)
    #expect(tuner.current == 2)
    tuner.observe(accepted: 0, drafted: 2)
    #expect(tuner.current == 1)
    tuner.observe(accepted: 0, drafted: 1)
    #expect(tuner.current == 1)
}

@Test func draftTokenTunerRespectsBounds() {
    var tuner = DraftTokenTuner(initial: 10, maximum: 3)

    #expect(tuner.current == 3)
    tuner.observe(accepted: 3, drafted: 3)
    tuner.observe(accepted: 3, drafted: 3)
    #expect(tuner.current == 3)
}
