# Roadmap

Planned work. Not a support claim.

## Completion Order

Ship finished model packages incrementally. Do not wait for a whole family, distributed lane, or
diffusion lane before uploading a package whose evidence and public card gates already pass.

Keep `main` clean as each feature slice finishes: group related work into reviewable commits, run
the relevant no-load gates before each commit, and run the full local publication gate before any
upload.

Order of work:

1. Single-device text model packages: close runtime parity, determinism, and practical full native
   context gates per package. Hybrid Qwythos/qwen3_5 remains gated until the 1,048,576-token replay
   finishes within the practical threshold; native KV allocation alone is not enough for a public
   context label.
2. Multi-device/staged model packages: prove package-specific staged parity first, then installed
   two-machine caix execution. Do not use generic POC evidence for package promotion.
3. Multimodal media blocks: keep packages moving as their exact runtime evidence closes, then add
   audio, video, and multipart request handling behind parser/runtime tests. Public cards name only
   the modalities with runtime serving evidence for that exact package.
4. Distributed inference: complete the two-machine pipeline path before any transport-speed or
   memory-scaling language. RDMA/TB5 remains design-only until hardware evidence exists.
5. Model uploads: each finished package gets the simple card contract: small RHM logo, install/run
   commands, base model, context, quantization/precision, download size, short status, license, and
   the open-source support link. Build process, test-device details, and internal evidence stay out
   of cards.
6. Diffusion: last. Return to diffusion only after the currently achievable single-device,
   multi-device, multimodal, and distributed beta packages have shipped or have explicit blocked
   evidence.

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

1. Prove same-machine staged execution and per-stage KV handling.
2. Verify staged output against a stable oracle before treating transport evidence as release
   evidence.
3. Prove Brew-installed two-machine transport on the MacBook.
4. Add Thunderbolt Bridge or other direct-link measurements before making link-speed claims.
5. Add the 16 GB Mac mini as a third shard after the two-machine path remains repeatable.

Current program state: the temporary Studio-only/MacBook-unavailable overlay is superseded. The
Brew-installed `0.2.13-beta` release completed an on-device Gemma 4 image+text validation on the
32 GB MacBook, and a Studio-to-MacBook staged run produced a coherent real prompt answer. That closes
a functional two-machine transport POC, but it is not a throughput, memory-pooling, Thunderbolt
Bridge, or TB5/RDMA claim.

The canonical staged correctness gate still uses an admissible oracle. For fp16 staged assets, the
fallback HF/PyTorch oracle is teacher-forced so tie flips do not cascade. The Qwen3-0.6B
staged-vs-HF-fp16 gate validates prefill, decode, and per-stage KV over the 128-token prompt set:
1020/1024 steps matched exactly and 4/1024 were genuine fp16 ties inside the 0.02 logit tolerance,
with zero real divergences; the repo-local evidence record is
`docs/distributed-evidence/qwen3-0.6b-teacher-forced-fp16-128.txt`. Other staged artifacts still need
per-model gates before any 1:1 claim.

Stateful monolithic prefill remains a Core AI runtime/export limitation above the 16-token traced
query width. The shipping caix mitigation keeps stateful monolithic prefill chunked to 16 by default
under `MonolithicPrefillPolicy`, while explicit env overrides remain available. Apple issue #84 tracks
the upstream stateful-prefill nondeterminism; fast unsafe batch prefill is not a release path.
Monolithic labels must distinguish deterministic 4-bit greedy output from fp16 1:1 HF claims.

Gemma 4 image+text is now a staged runtime path with verified runtime bundle and serving evidence,
not an S2-only plan. The production fix is the PLE image-token PAD mask plus
split-cache/single-asset staged execution; Apple issue #83 tracks the upstream PLE exporter gap.
Published Gemma 4 multimodal staged packages cover E2B, E4B, 12B, 26B-A4B, and 31B at full native
context with production-style cards. Do not market these as monolithic fast-path packages; current
Core AI image+text handling depends on staged splice and attention-mask contracts.

Hybrid `qwen3_5` support is in bring-up. Qwythos split-state staging has passed the tiny mixed-state
Core AI proof, short HF-token parity, native 1,048,576 KV-capacity allocation, and contiguous replay
through 16,384 tokens. It is still not a public full-1M context claim because the measured replay
slope projects to about 13.1 h for the whole context; public sequential context is capped at 16,384
tokens until replay performance improves.
`Qwen3.6-40B-Deckard-Heretic-GGUF` is in the registry as a requested GGUF-only image-text candidate:
the first feasible gate is a text-only GGUF dequant smoke, while real image-text conversion needs a
matching safetensors/base checkpoint or a dedicated `mmproj` importer.

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
