# Changelog

## Unreleased

### Added

- Added server-side discovery for installable `redhillsmediafl/*-caix` Hugging Face model repos.
- Added `POST /api/rhm-download` to install already-converted RHM Core AI bundles into the local exports directory.
- Added persistent converter and support-check logs under `~/.caix/logs` and `~/.caix/support-logs`.

### Fixed

- Inferred a conservative qwen3_5 hybrid KV-cache floor for converted bundles that do not yet carry `language.min_kv_capacity`, preventing under-sized cache allocation for Ornith-style bundles.
