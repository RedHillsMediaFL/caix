# Changelog

## Unreleased

### Added

- Added server-side discovery for installable `redhillsmediafl/*-caix` Hugging Face model repos.
- Added `POST /api/rhm-download` to install already-converted RHM Core AI bundles into the local exports directory.
- Added persistent converter and support-check logs under `~/.caix/logs` and `~/.caix/support-logs`.
- Rebuilt the dashboard as a model-agnostic server console with simple and advanced modes, local model controls, RHM installs, arbitrary HF support checks, conversion jobs, server health, and a link to the dedicated `/chat` page.
- Added persistent classic speculative serving for target+draft packages with nested `draft/` bundles, enabling the RHM Qwen MTP package to run as MTP instead of target-only.
- Added EAGLE/MTP serve and CLI flags for vocabulary, hidden size, sliding window, and max context so speculative targets are not hardcoded to one Gemma size.
- Exposed speculative runtime dimensions in the dashboard's advanced server panel.
- Added local discovery/loading and RHM installability for EAGLE target+draft package directories.
- Added a user-focused usage dashboard with rolling tok/s, last-generation speed, total input/output tokens, rolling-window output, and visible per-model throughput.
- Added a one-line GitHub installer that clones or updates `~/caix`, builds the Core AI runtime binary, and links the `caix` launcher into `~/.local/bin`.
- Added `scripts/refresh-export-index.sh` and `CAIX_EXPORT_INDEX` support for launchd/external-volume model discovery.
- Added an OpenCode provider config that points OpenCode at the local OpenAI-compatible caix server.
- Expanded the OpenCode provider config to expose the current caix bundle IDs for hot-loading through `/v1/chat/completions`.
- Added README status badges for GitHub releases, stars, Apple silicon/Core AI, Swift, and RHM Hugging Face models.
- Published verified RHM GLM-4-9B-0414 and GLM-4-32B-0414 Core AI bundles and documented them in the README model table.
- Published verified RHM Mixtral-8x7B-Instruct-v0.1 Core AI bundle and documented it in the README model table.
- Documented the verified RHM GPT-OSS 20B Core AI bundle in the README model table.
- Added structured Core AI authoring requirements for `qwen3_5_moe` support checks, covering the larger Ornith and Qwen3.6 MoE lane.
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

### Fixed

- Fixed server-side CoreAILanguageModels generation stalls by pumping the main runloop while `serve` waits on the HTTP server task.
- Fixed `convert.py <registry-key> --check` so it resolves registry keys to their Hugging Face repo before probing support.
- Generalized converter chat-template postprocessing so Qwen3.5 hybrid/MoE exports with `enable_thinking` branches start OpenAI output in visible content.
- Defaulted standard language-bundle serving to the stable one-shot CoreAILanguageModels path, with the older sequential engine available through `COREAI_LEGACY_ENGINE=1`.
- Pointed the OpenCode default model at the installed `qwen3-0.6b-coreai` bundle so `opencode run` works on the current local server while larger bundle IDs remain available in the provider map.
- Inferred a conservative qwen3_5 hybrid KV-cache floor for converted bundles that do not yet carry `language.min_kv_capacity`, preventing under-sized cache allocation for Ornith-style bundles.
- Inferred EAGLE target hidden size from Core AI model descriptors at load time, so larger targets can override the 26B default without source changes.
- Updated diffusion denoiser tests to match the official entropy-bound sampler behavior already implemented in the runtime.
- Listed all accepted EAGLE serve flags in CLI help.
- Kept dashboard model listing responsive under launchd by using bounded model-index and registry reads instead of blocking indefinitely on inaccessible export paths.
- Bounded dashboard Hugging Face support checks so a launchd/external-volume converter hang returns JSON and writes a support log instead of wedging the API.
- Defaulted raw HF `glm4` conversions to bfloat16 after support detection and logged not-yet-authored or failed converter attempts to `~/.caix/convert-failures.log`.
- Accepted `CAIX_COREAI_MODELS` as either a CoreAI checkout root or its `python/` package directory.
