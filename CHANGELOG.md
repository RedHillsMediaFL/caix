# Changelog

## Unreleased

### Added

- Added `caix --version`, `caix doctor`, a Core AI runtime checker, and a tap-ready Homebrew formula
  for the post-testing tap path.
- Added a release-version guard so caix stays below `1.0.0` while Core AI is beta.
- Added a version-sync guard for the CLI version, package script, and Homebrew formula.
- Added a Brew distributed-surface check for future Thunderbolt inference tests.
- Added a distributed readiness gate so Thunderbolt testing stays blocked until same-machine,
  loopback, Brew install, `cluster join`, and `serve --cluster` are ready.
- Added structured distributed evidence checks so same-machine and loopback token-match files
  cannot be placeholders and must carry raw-log and manifest fields.
- Tightened distributed evidence checks to require committed repo-relative manifest/raw-log paths
  and a `caix_commit` present in the local repository.
- Added a shared distributed stage-manifest loader so runtime code and `caix cluster plan` validate
  the same staged model contract.
- Added staged hidden-state boundary tensor metadata to the distributed manifest contract.
- Added boundary tensor validation to `DistributedStagePlan` packet checks.
- Added manifest-backed construction for the same-machine distributed pipeline harness.
- Added a manifest stage-handle factory path for future Core AI staged execution.
- Added stage-handle factory context with resolved stage asset URL checks.
- Added a stage-handle factory guard for missing resolved stage assets before Core AI stage load.
- Added typed distributed worker control/forward message frames for loopback and Thunderbolt
  transport work.
- Added canonical `DistributedStagePlan` integrity hashes for distributed worker handshakes.
- Added `HELLO` frame validation for distributed worker plan-hash, stage descriptor, and boundary
  contract checks.
- Added distributed allocation/reset/free control-frame validation for request IDs, stage IDs, and
  KV capacity.
- Added whole-message validation for distributed worker frames, including ACK and error frames.
- Added a JSON-line codec for distributed worker message headers.
- Added payload byte-count validation for distributed worker wire frames.
- Added full distributed worker wire-frame encoding and decoding for header-plus-payload transport.
- Added an incremental distributed worker wire-frame stream decoder for loopback socket reads.
- Added an in-process distributed loopback worker transport that exercises the same frame stream
  path planned for local worker sockets.
- Added a distributed worker frame executor that dispatches validated request frames to a stage
  handle and returns forward-result frames.
- Added a distributed worker handshake coordinator for `HELLO`/ACK admission and missing-stage
  startup checks.
- Added distributed worker request-state tracking for allocate-before-forward, step order,
  processed-token position, KV capacity, reset, and free handling.
- Tightened distributed worker `FREE` handling so unknown request IDs are rejected before stage
  teardown runs.
- Threaded `HELLO`/ACK through the loopback worker transport so handshake frames use the same
  stream decoder contract as worker requests.
- Added a remote distributed stage handle for coordinator-to-worker frame round trips.
- Tightened remote distributed stage handles so worker `ERROR` frames preserve code/detail and
  reject mismatched request or stage envelopes.
- Converted distributed loopback worker execution failures into validated `ERROR` frames so the
  in-process transport matches the socket contract.
- Tagged distributed runtime validation failures as `runtime_validation` worker `ERROR` frames
  instead of generic worker errors.
- Routed benchmark heavy-job preflights through `scripts/conversion-guard.sh` so lock/process
  detection stays in one place.
- Redacted sensitive-looking values from `scripts/conversion-guard.sh` process reports before text
  or JSON output.
- Added `git-lfs`, wrapped `git lfs`, and legacy `huggingface-cli` transfer detection to the
  shared heavy-job guard.
- Rejected zero or malformed `scripts/conversion-guard.sh --wait --interval` values.
- Redacted URL-embedded credentials from `scripts/conversion-guard.sh` process reports.
- Redacted sensitive URL query parameters from `scripts/conversion-guard.sh` process reports.
- Treated direct `curl`, `rsync`, `tar`, and `zip` processes as heavy activity in the shared guard.
- Added same-machine pipeline guards for invalid position ranges and reset request IDs.
- Added the cluster-plan contract check to publication gates.
- Added fail-closed `caix cluster join` and `caix serve --cluster` CLI stubs for Brew surface tests.
- Expanded the Homebrew formula test to cover distributed plan/join help and `serve --cluster`.
- Expanded Brew release checks to validate the staged Qwen3 distributed manifest and exact caix
  version through the package script and formula test.
- Let the Homebrew formula install packaged release tarballs with `bin/caix` while keeping source
  builds for `--HEAD`.
- Added a same-machine distributed stage-handle harness with fake-stage tests.
- Published verified RHM Qwen2.5-0.5B-Instruct and Qwen2.5-3B-Instruct Core AI bundles and
  documented them in the README model table.
- Published verified RHM Qwen3-8B Core AI bundle and documented it in the README model table.
- Published verified RHM Qwen3-14B Core AI bundle and documented it in the README model table.
- Published verified RHM Mistral-7B-Instruct-v0.3 Core AI bundle and documented it in the README
  model table.
- Published verified RHM Mistral-Nemo-Instruct-2407 Core AI bundle and documented it in the README
  model table.
- Published verified RHM Mistral-Small-Instruct-2409 Core AI bundle under the Mistral Research
  License and documented it in the README model table.
- Added a benchmark protocol and raw-log runner so future public speed numbers are reproducible.
- Added a benchmark suite manifest so every known RHM caix repo is measured or skipped with a stated
  reason.
- Added a benchmark-suite guard that refuses measured rows without an exact model repo revision.
- Added a benchmark gap audit that reports eligible manifest rows without committed measured raw
  evidence.
- Added a tracked-evidence mode for benchmark raw checks and enabled it in publication gates.
- Added direct benchmark-runner guards that require model repo id and exact repo revision.
- Added benchmark runner guards that refuse measured runs from a dirty git worktree.
- Added benchmark runner guards for active resumable Hub payload upload jobs.
- Expanded `scripts/conversion-guard.sh` to honor the heavy-task lock and active Hub transfer,
  build, test, benchmark, and verification jobs.
- Added direct benchmark-runner checks for stable measured stdout and generated-token counts.
- Added standalone benchmark-runner seed support for reproducible caix runs.
- Documented benchmark seed as a comparable run setting.
- Added a suite preflight that rejects seeded EAGLE/MTP benchmark rows.
- Added a benchmark report gate that refuses missing raw logs and marks rows without model revisions
  as non-publishable.
- Added a single publication-gates script for local and Hub metadata/card checks.
- Added whitespace and shell syntax checks to the publication-gates script.
- Added a cleanup-safety gate so export removal keeps respecting dry-run, lock, and bundle-name
  guards.
- Added an export-cleanliness gate so publication checks fail if local model payloads are left
  under `models/exports`.
- Added external tester instructions for verified load/generation reports and raw benchmark
  submissions.
- Added a public-copy guard script for benchmark placeholders, hype, unsupported support-product
  wording, and vague model-size language.
- Added a benchmark coverage guard that compares the manifest with live RHM caix repos on Hugging
  Face.
- Added a Hugging Face collection guard that checks public family collections cover the manifest and
  keep notes free of speed/fluff wording.
- Added a Hugging Face model-card guard that checks live manifest README files without downloading
  model payloads.
- Added a conversion ledger and guard so active registry lanes are marked published,
  component-only, or blocked with an explicit next step.
- Added classic speculative and EAGLE/MTP benchmark runner support so target+draft packages can
  produce raw benchmark logs instead of staying manual-only.
- Added benchmark-mode fields to suite summaries and reports so standalone, classic speculative,
  and EAGLE measurements stay separate.
- Canonicalized EAGLE benchmark rows as `eagle-mtp`, guarded the manifest mode values, and kept
  older `eagle` suite summaries readable.
- Added a reusable disk-pressure guard and wired benchmark runners to fail before writing logs when
  the checked volume is below the configured free-space floor.
- Added a disk-pressure guard check to publication gates so low-space preflights are exercised.
- Added a disk preflight for dashboard RHM bundle downloads before launching `hf download`.
- Added a disk preflight for dashboard HF and GGUF conversions before launching `convert.py`.
- Let benchmark suite runs pass exact model repo revisions into raw logs through a revisions TSV.
- Added a metadata-only Hugging Face revision collector for benchmark manifests.
- Added a generated tester request sheet for model verification and raw benchmark collection.
- Added `caix catalog` for metadata-backed Hugging Face caix repo discovery.
- Added exact Hub revisions to `caix catalog` install commands when the metadata includes a SHA.
- Pinned dashboard RHM downloads to the discovered Hub revision and defaulted direct bundles to the metadata bundle name.
- Defaulted converter and support-check HF cache placement to the checkout parent volume to reduce
  boot-disk writes during HF conversions.
- Added explicit text-only errors for OpenAI/Anthropic requests that include multimodal content blocks.
- Added server-side discovery for installable `redhillsmediafl/*-caix` Hugging Face model repos.
- Added `POST /api/rhm-download` to install already-converted RHM Core AI bundles into the local exports directory.
- Added persistent converter and support-check logs under `~/.caix/logs` and `~/.caix/support-logs`.
- Rebuilt the dashboard as a model-agnostic server console with simple and advanced modes, local model controls, RHM installs, arbitrary HF support checks, conversion jobs, server health, and a link to the dedicated `/chat` page.
- Added persistent classic speculative serving for target+draft packages with nested `draft/` bundles, enabling the RHM Qwen MTP package to run as MTP instead of target-only.
- Added EAGLE/MTP serve and CLI flags for vocabulary, hidden size, sliding window, and max context so speculative targets are not hardcoded to one Gemma size.
- Exposed speculative runtime dimensions in the dashboard's advanced server panel.
- Added local discovery/loading and RHM installability for EAGLE target+draft package directories.
- Added a usage dashboard with rolling tok/s, last-generation speed, total input/output tokens, rolling-window output, and per-model throughput.
- Added a one-line GitHub installer that clones or updates `~/caix`, builds the Core AI runtime binary, and links the `caix` launcher into `~/.local/bin`.
- Added `scripts/refresh-export-index.sh` and `caix_export_index` support for launchd/external-volume model discovery.
- Added an OpenCode provider config that points OpenCode at the local OpenAI-compatible caix server.
- Expanded the OpenCode provider config to expose the current caix bundle IDs for hot-loading through `/v1/chat/completions`.
- Added README status badges for GitHub releases, Apple silicon/Core AI, Swift, and RHM Hugging Face models.
- Published verified RHM GLM-4-9B-0414 and GLM-4-32B-0414 Core AI bundles and documented them in the README model table.
- Published verified RHM Mixtral-8x7B-Instruct-v0.1 Core AI bundle and documented it in the README model table.
- Documented the verified RHM GPT-OSS 20B Core AI bundle in the README model table.
- Added structured Core AI authoring requirements for `qwen3_5_moe` support checks, covering the heavier Ornith and Qwen3.6 MoE lane.
- Added a lightweight stdlib support-check path so the dashboard can inspect HF configs without launching the external CoreAI Python checkout.
- Added a native server-side HF config probe for dashboard support checks so launchd does not need to spawn converter subprocesses for quick architecture inspection.
- Marked `qwen3_5_moe` as an authored conversion family after adding native Core AI support in the vendored model registry.
- Added `COREAI_PERSISTENT_FAST_ENGINE=1` as an opt-in for the experimental kept-hot CoreAILanguageModels server engine.
- Added `scripts/conversion-guard.sh` so local conversion queues can avoid overlapping `convert.py`, raw `coreai.llm.export`, and CLI verification jobs.
- Added `mixtral-8x7b-instruct-coreai` to the OpenCode provider map.
- Removed `qwythos-9b-coreai` from the OpenCode provider map until its qwen3_5 reasoning-only output is fixed.
- Added converter postprocessing for Qwythos chat templates so future qwen3_5 exports bake the model's no-thinking branch for OpenAI-visible content.
- Published verified RHM Qwythos-9B-Claude-Mythos-5-1M Core AI bundle and documented it in the README model table.
- Restored `qwythos-9b-coreai` to the OpenCode provider map after no-thinking template verification.
- Added a registry conversion lane for the fully cached Ornith-1.0-35B qwen3_5_moe checkpoint.
- Added a qwen3_5_moe-safe int8 compression YAML and `convert.py --compression-config` passthrough for the Ornith-1.0-35B lane after the generic int4 export crashed during Core AI dequantization.
- Recorded the Ornith-1.0-35B runtime block: int8 conversion completes, but the 32 GB bundle does not pass live smoke on the 64 GB host, so HF publication remains blocked.
- Recorded the next Ornith-1.0-35B support path: CoreAI fork commit `648ad274` adds opt-in authored qwen3_5_moe quantization, and a one-layer mixed dense-int4/expert-int8 probe passes GPU fast-path smoke while full-bundle size tuning remains open.
- Recorded the authored Ornith-1.0-35B int4+head-quant path: CoreAI fork commit `7eafd4d` quantizes the shared head, one-layer and four-layer structural bundles pass GPU fast-path smoke, and the full 40-layer export completes at 17 GB, but full runtime warmup currently fails with an MPS reshape error, so HF publication remains held.
- Narrowed the authored Ornith-1.0-35B runtime boundary: the single-function 14-layer int4+head-quant probe exposed a prefill/decode shape-specialization gate, and dual-entrypoint exports now pass 14-layer two-token and 16-layer eight-token sequential generation without Core AI reshape/slice diagnostics; full-bundle validation remains held before HF publication.
- Added `caix inspect --model <bundle>` to specialize exported Core AI assets and print their
  resolved function inputs, outputs, states, scalar types, and shapes for runtime-contract
  debugging.

### Changed

- Tightened sampler hot paths by scanning logits and candidate buffers without enumerator overhead.

### Fixed

- Fixed `caix serve` default port: the CLI bound `8080` while the README, launcher, installer, and
  `opencode.json` all use `1237`. The default is now `1237` so a bare `caix serve` matches the docs.
- Added explicit KV-cache input/output execution for unoptimized Core AI language exports whose
  functions expose `keyCache`/`valueCache` as regular inputs and outputs instead of Core AI states.
- Routed qwen3_5/qwen3_5_moe bundles with packed recurrent-state KV floors to the explicit Core AI
  sequential engine by default, because the CoreAILanguageModels fast warmup currently uses a cache
  shape too small for Ornith's 1024-position SSM prefix; `COREAI_FAST_HYBRID_ENGINE=1` can still
  opt into the experimental fast path.
- Stopped the sequential language engine from running an unused final decode forward after the
  requested `maxTokens` count has already been emitted, which lets the 14-layer Ornith-35B depth
  probe complete one-token generation and avoids extra decode-shape pressure on hybrid exports.
- Loaded optional `language.function_map` decode entrypoints in the sequential runtime and routed
  one-token forwards through them; added opt-in `COREAI_PREFILL_CHUNK`/`COREAI_PREFILL_MODE`
  diagnostics for prefill/decode shape investigation without changing the default batched prefill.
- Loaded optional `assets.decode` split decode `.aimodel` packages so heavier hybrid bundles can keep
  prefill and one-token decode as separate Core AI assets when a combined multi-function export
  exceeds host memory.
- Baked `language.min_kv_capacity` into future converted qwen3_5/qwen3_5_moe bundle metadata and
  corrected the Ornith-1.0-35B registry floor to 1024.
- Fixed server-side CoreAILanguageModels generation stalls by pumping the main runloop while `serve` waits on the HTTP server task.
- Fixed `convert.py <registry-key> --check` so it resolves registry keys to their Hugging Face repo before probing support.
- Generalized converter chat-template postprocessing so Qwen3.5 hybrid/MoE exports with `enable_thinking` branches start OpenAI output in visible content.
- Defaulted standard language-bundle serving to the stable one-shot CoreAILanguageModels path, with the older sequential engine available through `COREAI_LEGACY_ENGINE=1`.
- Pointed the OpenCode default model at the installed `qwen3-0.6b-coreai` bundle so `opencode run` works on the current local server while heavier bundle IDs remain available in the provider map.
- Inferred a conservative qwen3_5 hybrid KV-cache floor for converted bundles that do not yet carry `language.min_kv_capacity`, preventing under-sized cache allocation for Ornith-style bundles.
- Inferred EAGLE target hidden size from Core AI model descriptors at load time, so wider targets can override the 26B default without source changes.
- Updated diffusion denoiser tests to match the official entropy-bound sampler behavior already implemented in the runtime.
- Listed all accepted EAGLE serve flags in CLI help.
- Kept dashboard model listing responsive under launchd by using bounded model-index and registry reads instead of blocking indefinitely on inaccessible export paths.
- Bounded dashboard Hugging Face support checks so a launchd/external-volume converter hang returns JSON and writes a support log instead of wedging the API.
- Defaulted raw HF `glm4` conversions to bfloat16 after support detection and logged not-yet-authored or failed converter attempts to `~/.caix/convert-failures.log`.
- Accepted `caix_coreai_models` as either a CoreAI checkout root or its `python/` package directory.
