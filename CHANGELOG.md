# Changelog

## Unreleased

### Added

- Added server-side discovery for installable `redhillsmediafl/*-caix` Hugging Face model repos.
- Added `POST /api/rhm-download` to install already-converted RHM Core AI bundles into the local exports directory.
- Added persistent converter and support-check logs under `~/.caix/logs` and `~/.caix/support-logs`.
- Rebuilt the dashboard as a model-agnostic server console with simple and advanced modes, local model controls, RHM installs, arbitrary HF support checks, conversion jobs, server health, and a link to the dedicated `/chat` page.
- Added persistent classic speculative serving for target+draft packages with nested `draft/` bundles, enabling the RHM Qwen MTP package to run as MTP instead of target-only.

### Fixed

- Inferred a conservative qwen3_5 hybrid KV-cache floor for converted bundles that do not yet carry `language.min_kv_capacity`, preventing under-sized cache allocation for Ornith-style bundles.
