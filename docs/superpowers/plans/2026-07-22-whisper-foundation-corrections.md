# Whisper Foundation Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close every native Whisper large-v2 foundation review blocker without loading or converting the full model.

**Architecture:** Keep the fixed split model, but move its contract to `caix.whisper-split.v2` with explicit Int32 load/decode statuses and tensor-gated state mutation. Authenticate all host generation rules and authoring inputs at their consumption boundaries, stage a strict CAIX provenance manifest beside the CoreAI files, validate it before load or promotion, and reject peak-memory/disk failures before a candidate becomes visible.

**Tech Stack:** Swift 6, XCTest, Python 3.12, pytest, PyTorch 2.9, CoreAI b2, descriptor-first POSIX file I/O, SHA-256, APFS atomic rename/swap.

## Global Constraints

- Work only in `/Volumes/SSD/caix/.worktrees/gemma4-31b-whisper-large-v2`.
- Preserve the pinned Hugging Face cache, the existing 3 GB asset, and the exact authoring environment.
- Do not run the full export or full saved-model load without controller release.
- Use strict red-green TDD for every behavior change.
- Reject timestamps instead of partially implementing timestamp rules.
- Keep request audio, features, state, tokens, and transcripts memory-only.
- Publish only a fully authenticated candidate that remained below the configured recorded peak-RSS limit.
- Commit Swift safety/policy separately from Python ABI/provenance changes.

---

### Task 1: Swift fail-closed decoding policy and callback lifetime

**Files:**
- Modify: `Sources/PipelineRuntime/WhisperDecodingPolicy.swift`
- Modify: `Sources/PipelineRuntime/WhisperDecoderLoop.swift`
- Modify: `Sources/CoreAIServer/InMemoryAudioDecoder.swift`
- Modify: `Sources/CoreAIServer/WhisperLogMelExtractor.swift`
- Modify: `Package.swift`
- Modify: `Tests/PipelineRuntimeTests/WhisperDecodingPolicyTests.swift`
- Modify: `Tests/PipelineRuntimeTests/WhisperDecoderLoopTests.swift`
- Modify: `Tests/CoreAIServerTests/InMemoryAudioDecoderTests.swift`

**Interfaces:**
- Consumes: `ResidentModelLock.speech.metadata.generationConfigSHA256` and `BoundedRegularFileReader.read`.
- Produces: authenticated policy construction, `PolicyError.timestampsUnsupported`, zero-step timestamp rejection, post-callback cancellation, and a retained AudioToolbox callback context.

- [ ] **Step 1: Write failing policy/loop tests**

  Add tests which load the repository lock, authenticate the exact pinned generation config, reject a one-byte/suppression-map alteration, assert `includeTimestamps: true` throws `.timestampsUnsupported` before `step`, and cancel from the final streaming callback.

- [ ] **Step 2: Verify the Swift tests fail for the missing behavior**

  Run:

  ```bash
  swift test --filter 'WhisperDecodingPolicyTests|WhisperDecoderLoopTests'
  ```

  Expected: failures for the missing authenticated initializer, missing typed timestamp error, nonzero timestamp step count, or missing final cancellation check.

- [ ] **Step 3: Implement authenticated policy construction and fail-closed loop behavior**

  Use a descriptor-first bounded read and SHA-256 comparison against the validated resident lock before decoding JSON. Keep fixture-only authenticated-data construction internal. Guard timestamps before the first step and call `Task.checkCancellation()` immediately after every callback and before returning.

- [ ] **Step 4: Write and fail an explicit callback-lifetime test**

  Exercise a retained callback-context helper with a weak reference/deinit probe, proving the context remains alive inside the callback and is released exactly after the protected scope.

- [ ] **Step 5: Implement the strong callback lifetime**

  Replace the unmanaged unretained lifetime with a scoped `passRetained`/`release` helper (or an equivalent explicit `withExtendedLifetime` scope) covering `AudioFileOpenWithCallbacks` through the last `ExtAudioFileRead`.

- [ ] **Step 6: Run focused Swift tests and commit**

  Enable Accelerate's supported ILP64 CBLAS import with the two documented Clang definitions,
  update the bounded matrix dimensions to the imported integer type, and rerun the log-mel golden
  test to prove numerical behavior is unchanged.

  Run:

  ```bash
  swift test --filter 'WhisperDecodingPolicyTests|WhisperDecoderLoopTests|InMemoryAudioDecoderTests|WhisperLogMelExtractorTests'
  ```

  Expected: all selected tests pass with zero failures.

  Commit message: `fix(whisper): authenticate host decoding and retain audio callbacks`

### Task 2: ABI v2 exact native state machine

**Files:**
- Modify: `python/whisper_large_v2/abi.py`
- Modify: `python/whisper_large_v2/export.py`
- Modify: `python/whisper_large_v2/reference.py`
- Modify: `python/tests/test_whisper_native_abi.py`
- Modify: `python/tests/test_whisper_full_export.py`
- Modify: `python/tests/test_whisper_full_coreai_export.py`
- Modify: `python/tests/test_coreai_whisper_state_probe.py`

**Interfaces:**
- Produces: ABI schema `caix.whisper-split.v2`; `load_cross_kv -> load_status`; `decode_step -> logits, decode_status`; success status `[1]`; invalid status `[0]`; invalid operations leave every state tensor bit-identical.

- [ ] **Step 1: Write eager failing tests for invalid transitions**

  Add direct tiny-model tests for successful load, second load, decode-before-load, readiness other than exactly one, and positions `-1`/`448`. Snapshot all six state tensors and assert exact equality after every invalid operation.

- [ ] **Step 2: Verify eager tests fail**

  Run:

  ```bash
  python -m pytest -q python/tests/test_whisper_native_abi.py python/tests/test_whisper_full_export.py
  ```

  Expected: v1 schema/output assertions and invalid-state mutation tests fail.

- [ ] **Step 3: Implement tensor-gated v2 state transitions**

  Gate load with `cross_ready == 0`; add payload/readiness only under that scalar condition. Gate decode with `cross_ready == 1 && 0 <= position < 448`; clamp the physical index, write back the existing self-cache slot when invalid, increment position only when valid, return zeroed unusable logits plus status zero on invalid input, and retain exact status one on success.

- [ ] **Step 4: Update CoreAI entrypoint names and tiny proof assertions**

  Register `decode_status` as the second decoder output, test saved CoreAI success/invalid behavior, and compare every invalid pre/post state array with `np.array_equal`.

- [ ] **Step 5: Run eager tests and one memory-approved tiny CoreAI smoke**

  Run eager tests first. If available memory is green, run only the gated minimal CoreAI test; never set the full-asset/full-checkpoint gates.

- [ ] **Step 6: Commit**

  Commit message: `fix(whisper): enforce the v2 native state machine`

### Task 3: Pinned authoring source and descriptor-bound config

**Files:**
- Create: `models/coreai-models-authoring-source.json`
- Create: `python/whisper_large_v2/authoring_source.py`
- Modify: `python/whisper_large_v2/checkpoint.py`
- Modify: `python/whisper_large_v2/convert.py`
- Modify: `python/whisper_large_v2/export.py`
- Modify: `scripts/export-whisper-large-v2-coreai.sh`
- Modify: `python/tests/test_whisper_large_v2_checkpoint.py`
- Modify: `python/tests/test_whisper_full_checkpoint_export.py`

**Interfaces:**
- Produces: pinned `coreai-models` repository/revision/subtree identity, an authenticated archive-backed import root, and `load_verified_json_asset(snapshot, contract, "config.json")`.

- [ ] **Step 1: Write failing source-authentication tests**

  Test wrong revision/tree, a dirty working-tree replacement, import outside the authenticated root, and a config path replacement after initial snapshot validation.

- [ ] **Step 2: Verify the tests fail**

  Run the two checkpoint/export test files without real-source/full-load gates.

- [ ] **Step 3: Implement pinned CoreAI-models bootstrap**

  Resolve the exact commit and `python/src/coreai_models` Git tree from the tracked lock, archive that commit into a task-scoped temporary directory, prepend only the extracted `python/src`, and verify imported modules resolve below that root. Remove the mutable external working-tree `PYTHONPATH` entry from the export script.

- [ ] **Step 4: Consume config bytes through their authenticated descriptor**

  Read bounded JSON from `/dev/fd/<verified descriptor>` while the descriptor remains open and build `WhisperConfig.from_dict`; do not call `from_pretrained(snapshot)`.

- [ ] **Step 5: Run focused tests and commit**

  Commit message: `fix(whisper): pin authoring source and bind config reads`

### Task 4: Strict embedded provenance and safe publication

**Files:**
- Create: `python/whisper_large_v2/manifest.py`
- Modify: `python/whisper_large_v2/convert.py`
- Modify: `python/whisper_large_v2/verify.py`
- Modify: `python/tests/test_whisper_full_checkpoint_export.py`
- Modify: `python/tests/test_whisper_full_asset.py`

**Interfaces:**
- Produces: `caix-manifest.json`; streaming `sha256_regular_file`; `validate_caix_asset`; disk preflight; peak-RSS pre-publication hook; atomic sibling candidate save; APFS swap-based recoverable promotion.

- [ ] **Step 1: Write failing strict-manifest tests**

  Cover canonical successful validation plus absent/extra fields, extra directory entries, malformed JSON, symlinks, non-regular files, mismatched `main.hash`, mismatched manifest SHA-256, and altered pinned source/ABI/stack fields.

- [ ] **Step 2: Verify manifest tests fail**

  Run only pure Python tests over kilobyte-sized fake assets.

- [ ] **Step 3: Implement bounded manifest writing and validation**

  Stream-hash `main.mlirb`, compare the digest with the raw 32-byte `main.hash`, write a canonical strict sidecar containing the exact v2 ABI/source/stack and MLIR size/hash, fsync it, then revalidate all four asset files before promotion.

- [ ] **Step 4: Write failing disk/RSS/publication fault-injection tests**

  Inject insufficient disk, save failure, hash failure, manifest-write failure, peak-RSS excess, candidate-validation failure, and promotion failure. Assert hidden staging cleanup and byte-for-byte preservation of any prior final asset.

- [ ] **Step 5: Implement preflight and recoverable promotion**

  Require conservative free space before model materialization. Compare recorded process peak RSS before graph save and again inside the pre-publication callback. Refuse publication on excess. Save to an absent sibling candidate; validate it; use atomic rename for a missing final or APFS `RENAME_SWAP` for an existing final so the old asset remains at the candidate path until the swap succeeds.

- [ ] **Step 6: Run pure tests and commit**

  Commit message: `fix(whisper): authenticate artifacts before atomic publication`

### Task 5: Verification coverage and contract documentation

**Files:**
- Modify: `python/whisper_large_v2/verify.py`
- Modify: `python/whisper_large_v2/export.py`
- Modify: `python/tests/test_whisper_full_asset.py`
- Modify: `python/tests/test_whisper_full_coreai_export.py`
- Modify: `docs/WHISPER_LARGE_V2_NATIVE_ABI.md`

**Interfaces:**
- Consumes: v2 statuses and authenticated manifest.
- Produces: pre-load authentication; successful cross-cache equality; nonzero current self-cache slot; zero unused tail; invalid-transition immutability; accurate recorded-peak terminology.

- [ ] **Step 1: Write failing verifier/proof assertions**

  Require manifest validation to finish before `AIModel.load`; compare loaded cross caches to encoded payloads; require current self K/V slot mutation; assert decode-before-load/second-load/invalid-position status zero and exact state equality.

- [ ] **Step 2: Verify tests fail for the missing fields/assertions**

  Run pure verifier unit tests and the eager tiny proof only.

- [ ] **Step 3: Implement the strengthened verifier and tiny proof**

  Return explicit dtype/status/cache-error/current-slot metrics and reject any mismatch. Keep hashing streaming and state comparisons bounded to the already allocated fixed state.

- [ ] **Step 4: Update every v1 consumer and document the unsupported timestamp boundary**

  Replace `caix.whisper-split.v1` with v2 only where the output contract changed, document status values and invalid-state immutability, remove the hard-cap wording, and describe candidate/promotion recovery.

- [ ] **Step 5: Run fresh bounded verification**

  Run all Swift tests, all ungated Whisper Python tests, `git diff --check`, and inspect `git status`. Do not run full export/load gates.

- [ ] **Step 6: Commit**

  Commit message: `test(whisper): prove v2 provenance and state safety`

## Final Self-Review

- [ ] Re-read `.superpowers/sdd/whisper-foundation-corrections.md` and map every requirement to a passing test or explicitly reported controller-held step.
- [ ] Confirm no `includeTimestamps: true` path can call the decoder.
- [ ] Confirm invalid native transitions preserve every state tensor exactly.
- [ ] Confirm no mutable external `coreai_models` checkout is imported by the full export script.
- [ ] Confirm manifest validation precedes CoreAI model load and publication.
- [ ] Confirm recorded peak RSS is checked before publication and terminology never calls it an OS hard limit.
- [ ] Confirm the existing full asset, HF cache, external authoring environment, and unrelated worktrees are untouched.
