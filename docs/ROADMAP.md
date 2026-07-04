# Roadmap

Planned work. Not a support claim.

## Core AI Distributed Execution

Goal: run models that do not fit on one Mac by splitting Core AI execution across Macs.

Target hardware:

- Main Mac: coordinator and highest-memory stage.
- 32 GB MacBook: middle stage worker.
- 16 GB Mac mini: small stage worker, install tests, low-memory compatibility.

First implementation:

- Pipeline parallelism across Core AI stage bundles.
- One model split into exported stage bundles: embeddings, layer ranges, final norm/head.
- Each worker loads only its assigned stage.
- KV cache stays local to the worker that owns those layers.
- Hidden-state activations move between workers over Thunderbolt Bridge or LAN.
- Final stage samples and returns token IDs.

Current in-tree pieces:

- `DistributedStagePlan` and validation for roles, layer coverage, workers, and hidden-state
  packet routes.
- `DistributedStageManifest`, a shared loader for staged manifest and `metadata.json` cluster
  blocks, including hidden-state boundary tensor metadata.
- `caix cluster plan` dry-run placement for staged manifests.
- Minimal `caix cluster join` and `caix serve --cluster` staged-worker runtime.
- `DistributedSameMachinePipeline`, an in-process stage-handle harness tested with fake stages,
  manifest-ordered handle maps, and a stage-handle factory context with resolved stage asset paths.
- Typed worker message frames for hello/ack, allocation, forward, reset, free, and error.
- SHA-256 plan integrity hashes for worker handshakes.
- Worker frame execution, handshake admission, in-process loopback framing, and request-state
  guards for allocate-before-forward, step order, processed-token position, KV capacity, reset,
  and free.
- Core AI distributed stage handle pieces for descriptor validation, allocation, NDArray IO,
  output readback, and `.none`/`.stateful`/`.explicitOutputs` forward/reset execution.

Still missing before real distributed Qwen release:

- Per-model staged parity against an admissible oracle for each released staged artifact. Qwen3-0.6B
  noopt-first3 is closed against the teacher-forced HF-fp16 oracle; other staged Qwen artifacts still
  need their own gate before any 1:1 claim.
- Thunderbolt Bridge test evidence.

Why this path:

- Core AI exposes local `AIModel` / `InferenceFunction` execution with caller-owned input buffers,
  output views, and mutable KV state.
- The current public Core AI language runtime is local-device only.
- MLX/exo prove multi-Mac inference is practical, but caix needs a Core AI-native path for
  converted `.aimodel` bundles.

Do first:

1. Prove same-machine staged execution with Qwen3-0.6B.
2. Verify staged output against a stable oracle before treating transport evidence as release
   evidence.
3. Prove the tiny random Qwen3 staged POC over two machines through Brew when the MacBook is
   available again.
4. Move one real Qwen stage to the 32 GB MacBook over Thunderbolt Bridge.
5. Add the 16 GB Mac mini as a third shard.

Temporary Studio-only overlay, 2026-07-03: the MacBook is unavailable, so steps 3-5 are deferred.
P0 same-machine staged parity is now closed for Qwen3-0.6B noopt-first3 against the teacher-forced
HF-fp16 oracle. The local-only monolithic default-chunk validation and labeling/docs cleanup pass is
complete in the worktree and ready for review; multimodal S2, heavy exports, uploads, and public
speed/card updates stay held until explicit sign-off. This is a reviewed-ready local patch, not a
published release or HF card update.
No two-machine ready-to-test wording follows from this overlay. The canonical P0 gate must use an
admissible oracle: `scripts/run-staged-parity-p0.sh` proves
monolithic Core AI determinism across fresh processes before running staged-vs-monolithic parity,
and deterministic HF/PyTorch token sequences are the fallback staged-correctness oracle when that
monolithic pre-check fails. For fp16 staged assets, that fallback is teacher-forced so tie flips do
not cascade. The current 128-token Qwen3-0.6B staged-vs-HF-fp16 gate validates prefill, decode, and
per-stage KV: 1020/1024 steps matched exactly and 4/1024 were genuine fp16 ties inside the 0.02 logit
tolerance, with zero real divergences; the repo-local evidence record is
`docs/distributed-evidence/qwen3-0.6b-teacher-forced-fp16-128.txt`. Production qwen3-4b monolithic
determinism is a separate bug track: the production-config bundle is stable at 8/16-token prefills
and nondeterministic starting at 17 through both `LLMEngine` and the serving-compatible
`CoreAIPipelinedEngine` path. The caix-side
determinism mitigation is review-ready and locally validated in the worktree: stateful monolithic
prefill defaults to the traced query width 16, while explicit env overrides remain available.
Evidence under
`.tmp/benchmark-window/20260703T163939Z-qwen3-4b-prefill-mitigation/` shows baseline nondeterminism
at lengths 17/19/24, deterministic byte-identical chunk16 and token-mode outputs, and below-onset
len8/len16 divergence from HF-fp16, which proves the remaining HF split is 4-bit quantization rather
than a chunking bug. Tier-1 vendored export changes are shelved as unnecessary for this shipping fix.
The serving-path label is settled for review: `<model>-monolithic` is the single-device fast path,
deterministic by default under `MonolithicPrefillPolicy`, but 4-bit bundles must not claim fp16 1:1;
`<model>-staged` is the distributed/multi-device path, with Qwen3-0.6B fp16 verified 1:1 against HF
under the teacher-forced gate and per-model parity required before any other 1:1 claim. Same-machine
staged is allowed for correctness/debug placement, but it is not a single-device fast path. No
qwen3-4b staged artifact is present locally, and the held qwen3-4b staged export is shelved as
unnecessary for the serving-path decision unless BOSS explicitly reopens it. Existing Qwen3-0.6B
artifacts provide only internal directional local evidence: monolithic `171.3` tok/s median vs
same-machine staged `38.2` tok/s median over the captured 256-token probe (`R=0.223`). Do not publish
that as a model-card benchmark.
Current source-read root cause: vendored `coreai-models` traces the monolithic stateful export at
the 16-token prefill shape. `_constants.py:15` sets `TRACE_KV_CACHE_SEQ_LEN = 2048`, but that only
sizes the traced/reference state capacity. The 17-token onset lines up with `_constants.py:18`
`QUANT_TRACE_QUERY_LEN = 16`, which `export/macos.py:86-97` uses for `[1,16]` `input_ids` plus a
2048-token reference cache before exporting `keyCache`/`valueCache` as mutable Core AI state at
`export/macos.py:234-245`. The Python model-def itself remains runtime-shaped:
`models/macos/qwen3.py:83-100` derives `seq_len`, `query_len`, and `offset` from runtime
`position_ids`, and `primitives/macos/cache.py:115-164` writes/fetches KV with `mutable_slice_update`
plus `narrow(..., 0, seq_len)`. A no-load `torch.export` probe keeps the KV update end and fetch
length dynamic, so the fault surface is not Qwen3 model-def and not the 2048 cache bound; it is the
post-FX Core AI stateful export/lowering/runtime specializing the mutable-state write extent to the
16-token trace while attention/fetch follows runtime `position_ids`. The upstream/exporter fix would
make monolithic prefill/decode entrypoints or stateful prefill tracing genuinely dynamic, but the local
caix mitigation keeps the production query width within the traced boundary by default. Monolithic
labels must still distinguish deterministic 4-bit greedy output from fp16 1:1 HF claims.
Multimodal S2 now has deterministic media fixture generation plus a
no-load structural preflight in the exporter
workspace (`models/gemma4-unified-mm/run_s2_parity.py fixtures`,
`models/gemma4-unified-mm/run_s2_parity.py preflight`, `models/gemma4-unified-mm/run_s2_parity.py plan`).
The verified fixture set is 25 image, 10 audio, and 5 video prompts; preflight and plan artifacts are
not logit/token evidence.

API surface side quest, 2026-07-03: OpenAI `response_format` now decodes `text`, `json_object`, and
`json_schema` and carries the request into the internal generation contract. Runtime builds route
structured requests through CoreAILM `ConstrainedDecodingStrategy`. Normal text generation stays on
the fast pipelined engine; constrained requests use CoreAILM's sequential engine variant because it
exposes the per-step logits the grammar sampler needs. The qwen3-4b persistent-handle smoke passed
against the existing local bundle and produced schema-valid JSON at
`.tmp/structured-output-smoke/20260703T182811Z/qwen3-4b-coreai-persistent.json`. No-load APITypes
regression tests cover request mapping plus build/backend rejection. Keep public capability copy
gated until the release target has this smoke evidence, dependency provenance, and publication
sign-off. `scripts/check-structured-output-evidence.sh` now validates captured smoke JSON without
loading a model, and `scripts/check-structured-output-evidence-contract.sh` fixture-tests that
validator inside publication gates. Release
evidence now records the exact resolved `coreai-models`/xgrammar/Core AI Python dependency set through
`benchmarks/DEPENDENCY_EVIDENCE.tsv` and `scripts/check-dependency-evidence.sh` before public
capability claims.
Public `response_format` copy remains held on one reproducibility blocker: xgrammar is currently
recorded as `branch main` in `benchmarks/DEPENDENCY_EVIDENCE.tsv`. The no-load readiness command
`scripts/check-structured-output-release-readiness.sh --evidence <smoke.json>` validates smoke
evidence plus dependency provenance and fails strict mode until xgrammar is pinned or vendored.

No-load observability side quest, 2026-07-03: `/api/activity.prefixHitCount` is surfaced in the CLI
dashboard, ChatTUI `/activity`, and the web activity panels so exact-continuation prefix reuse can
be audited from normal operator views. The web dashboard also renders `ServerInfo.caixVersion` and
wires `/api/supported` into the Advanced Server panel. These are source/UI verification improvements,
not parity, performance, model-quality, or release-readiness evidence.

Tiny MacBook POC gate:

- Use `qwen3-tiny-random-coreai-staged-rope-input-f16-2x1`.
- Test the release path through Brew before running the smoke.
- Verify both machines and link speed with `caix deploy verify`.
- Verify staged bundle copy digests on the MacBook.
- `scripts/check-distributed-readiness.sh --tiny-poc --tiny-manifest <manifest> --brew-caix "$(command -v caix)"` must pass first.
- Real Qwen3-0.6B stays unpublished until token parity and load gates pass.

Do not start with:

- Tensor parallelism.
- All-reduce collectives.
- Shipping logits between machines.
- Claims that Mac memory is pooled.

The claim, once working: stage-sharded Core AI execution across Macs.
