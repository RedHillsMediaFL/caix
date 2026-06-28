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
- Added README status badges for GitHub releases, stars, Apple silicon/Core AI, Swift, and RHM Hugging Face models.
- Published verified RHM GLM-4-9B-0414 and GLM-4-32B-0414 Core AI bundles and documented them in the README model table.

### Fixed

- Inferred a conservative qwen3_5 hybrid KV-cache floor for converted bundles that do not yet carry `language.min_kv_capacity`, preventing under-sized cache allocation for Ornith-style bundles.
- Inferred EAGLE target hidden size from Core AI model descriptors at load time, so larger targets can override the 26B default without source changes.
- Updated diffusion denoiser tests to match the official entropy-bound sampler behavior already implemented in the runtime.
- Listed all accepted EAGLE serve flags in CLI help.
- Kept dashboard model listing responsive under launchd by using bounded model-index and registry reads instead of blocking indefinitely on inaccessible export paths.
- Bounded dashboard Hugging Face support checks so a launchd/external-volume converter hang returns JSON and writes a support log instead of wedging the API.
- Defaulted raw HF `glm4` conversions to bfloat16 after support detection and logged unsupported/failed converter attempts to `~/.caix/convert-failures.log`.
