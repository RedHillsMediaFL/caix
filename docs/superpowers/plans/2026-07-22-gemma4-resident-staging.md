# Gemma 4 Resident Staging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the authenticated Gemma 4 31B Q4 target with all decode assets resident while streaming at most one prefill asset at a time, under live memory-pressure admission.

**Architecture:** Parse and fail closed on the producer's exact `runtime_memory` contract. A residency-aware same-machine pipeline owns decode-resident stage handles and wraps every multi-token stage forward in a checked prefill lease; each concrete CoreAI handle retains its request KV state while its transient prefill model/function is released. `TextStagedModel` opts into this path only for the exact manifest policy and uses the already authenticated July Gemma 4 resident renderer.

**Tech Stack:** Swift 6, CoreAI direct runtime, swift-transformers, swift-jinja, XCTest.

## Global Constraints

- Start from commit `7f885f7`; do not touch conversion output or load full model assets.
- Honor `asset_residency_policy=decode_set_plus_one_streamed_prefill_stage` exactly.
- When `full_prefill_and_decode_sets_may_not_be_dual_resident=true`, never retain all prefill and decode models simultaneously.
- Decode models and request KV state remain resident; one prefill model is leased, run against that same state, then released before the next stage.
- Memory telemetry is checked before decode-resident startup and before every transient prefill load; yellow/red/unknown pressure and resident-limit decisions fail closed.
- Select the manifest's initial context only when its tier passes; otherwise select the declared fallback without allocating the rejected tier.
- Gemma 4 staged chat uses `Gemma4ChatTemplateContract.ResidentRenderer`, including rich messages, tools, reasoning fields, and additional context.
- Omitted API temperature resolves to greedy `0` only for staged target; explicit nonzero sampling remains rejected.
- `--prewarm all` may prewarm the memory-safe staged target while the legacy >=20B monolithic safeguard remains.

---

### Task 1: Authenticate the runtime-memory contract

**Files:**
- Modify: `Sources/PipelineRuntime/DistributedRuntime.swift`
- Test: `Tests/PipelineRuntimeTests/DistributedRuntimeTests.swift`

**Interfaces:**
- Produces: `DistributedRuntimeMemoryContract`, its initial/fallback tiers, and `DistributedStageManifest.runtimeMemory`.

- [ ] Add a manifest fixture containing both exact residency flags and initial/fallback byte/context tiers; assert all fields decode.
- [ ] Run `swift test --filter DistributedRuntimeTests/testStageManifestLoadsStreamedPrefillRuntimeMemoryContract` and observe failure because `runtimeMemory` does not exist.
- [ ] Decode the exact snake-case fields and validate the policy, tier ordering, positive byte counts, and distinct decode assets for every stage.
- [ ] Add rejection tests for a false dual-residency flag, malformed tiers, and missing decode assets.
- [ ] Run the focused manifest tests and commit `feat(gemma4): authenticate staged residency manifest`.

### Task 2: Enforce one transient prefill lease with live admission

**Files:**
- Create: `Sources/PipelineRuntime/Gemma4StagedResidency.swift`
- Create: `Tests/PipelineRuntimeTests/Gemma4StagedResidencyTests.swift`
- Modify: `Sources/PipelineRuntime/DistributedRuntime.swift`

**Interfaces:**
- Produces: `DistributedDecodeResidentStageHandle`, `DistributedStagedMemorySnapshot`, `DistributedStagedMemoryAdmission`, and residency-aware `DistributedSameMachinePipeline` construction.
- Consumes: `ResidentServiceHealthGate`, `DistributedRuntimeMemoryContract`.

- [ ] Write fakes that record `loadPrefill`, `forward`, and `unloadPrefill`, plus a deterministic memory snapshot provider.
- [ ] Assert a multi-token forward records `load -> forward -> unload` for each stage with maximum concurrent prefill residency of one; run and observe the missing API failure.
- [ ] Add the decode-resident handle protocol and pipeline lease wrapper with unconditional release on success, throw, and cancellation.
- [ ] Assert one-token decode never loads prefill and memory rejection happens before any load.
- [ ] Assert context selection chooses initial on a safe snapshot, fallback when initial does not fit, and rejects unsafe pressure without allocating either tier.
- [ ] Run `swift test --filter Gemma4StagedResidencyTests` and commit `feat(gemma4): serialize streamed prefill residency`.

### Task 3: Bind CoreAI decode-resident state to transient prefill

**Files:**
- Modify: `Sources/PipelineRuntime/DistributedCoreAIStageIOContract.swift`
- Test: `Tests/PipelineRuntimeTests/DistributedRuntimeTests.swift`

**Interfaces:**
- Produces: `DistributedCoreAIDecodeResidentStageHandleFactory` and a handle whose request-state dictionary is shared by resident decode and transient prefill functions.
- Consumes: `DistributedDecodeResidentStageHandle`.

- [ ] Add direct-runtime contract tests for rejecting absent/separate decode assets and incompatible prefill/decode descriptors.
- [ ] Run with `COREAI_DIRECT_RUNTIME=1` and observe failure before the factory exists.
- [ ] Specialize decode assets during factory creation, build request cache state from the decode descriptor, and retain no prefill model/function.
- [ ] On `loadPrefill`, specialize only that stage's main asset and validate identical state names/types/shapes; on `unloadPrefill`, clear every strong prefill reference.
- [ ] Route multi-token forwards through the leased prefill bindings and one-token forwards through the resident decode bindings while mutating the same request-state entry.
- [ ] Run direct-runtime compile and focused tests; commit `feat(gemma4): bind decode-resident CoreAI stages`.

### Task 4: Route staged Gemma serving through the resident contract

**Files:**
- Modify: `Sources/PipelineRuntime/TextStagedModel.swift`
- Modify: `Sources/CoreAIServer/APITypes.swift`
- Modify: `Sources/CoreAIServer/ModelManager.swift`
- Modify: `Sources/CoreAIServer/Server.swift`
- Test: `Tests/PipelineRuntimeTests/Gemma4ChatTemplateContractTests.swift`
- Test: `Tests/CoreAIServerTests/APITypesTests.swift`
- Test: `Tests/CoreAIServerTests/ModelManagerTests.swift`

**Interfaces:**
- Produces: rich staged `generate(messages:tools:additionalContext:options:onToken:)`, capability-aware temperature adjustment, and staged prewarm eligibility.

- [ ] Add a rich staged prompt test proving tool calls, reasoning content, and tool responses use the authenticated resident renderer rather than generic tokenizer templates.
- [ ] Retain one `ResidentRenderer` at staged-model load, add the rich overload, and make the string-only overload losslessly promote its current fields.
- [ ] Preserve `temperatureWasExplicit` in `GenerationRequest`; assert omitted temperature becomes `0` for `.textStaged` while explicit `0.7` remains `0.7` and is rejected by the backend.
- [ ] Add a >=20B `mode: "staged"` prewarm test and exempt only that mode from the legacy skip.
- [ ] Run focused PipelineRuntime/CoreAIServer tests and commit `feat(gemma4): wire resident staged serving`.

### Task 5: Verify integration without full assets

**Files:**
- Modify only files required by compiler/test findings.

- [ ] Run `swift test --filter 'DistributedRuntimeTests|Gemma4StagedResidencyTests|ResidentMemoryPlannerTests|APITypesTests|ModelManagerTests'`.
- [ ] Run `COREAI_DIRECT_RUNTIME=1 swift test --filter 'DistributedRuntimeTests|Gemma4StagedResidencyTests|Gemma4ChatTemplateContractTests|APITypesTests|ModelManagerTests'`.
- [ ] Run `git diff --check`, inspect every changed file, and confirm no conversion output or full model asset was accessed.
- [ ] Commit any verification-only correction with a detailed message and report exact integration commits.
