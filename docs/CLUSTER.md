# Cluster Planning

Planned feature. This page documents the first user-visible dry-run surface only.

`caix cluster plan` reads local staged-bundle metadata and prints a placement plan. It does not
load Core AI models, start workers, transfer tensors, benchmark, download, or upload anything.
JSON output includes a `runtime_plan` object using the same `DistributedStagePlan` contract that
the runtime validates, including `boundary_tensor`.

## Commands

```bash
caix cluster plan --manifest qwen3-stages.json --workers main=64,mbp=32,mini=16
caix cluster plan --model models/exports/qwen3-staged --worker main=64 --worker mini=16
caix cluster plan --manifest qwen3-stages.json --json
```

`--model` reads `metadata.json` from the bundle and expects a `cluster.stages` block. Current
single-bundle exports do not include that block, so the command reports a TODO telling the exporter
what metadata is missing.

The planner preserves stage order and assigns each stage to the first worker, in the order supplied,
with enough remaining memory. Worker memory is a dry-run budget in GB.

## Thunderbolt Test Gate

Do not ask for MacBook Thunderbolt testing yet. The gate is:

1. Same-machine staged Qwen3-0.6B matches the monolithic bundle token-for-token.
2. The same stage split works across two local processes over loopback.
3. `caix cluster join` and `caix serve --cluster` can run the same manifest used by the loopback test.

After those pass, test one MacBook stage over Thunderbolt Bridge. Until then, keep testing local.
Before the Thunderbolt test, install caix through Brew on the test machine and run
`scripts/check-publication-gates.sh --distributed --brew-caix "$(command -v caix)"`.

Use the readiness gate before asking for hardware:

```bash
scripts/check-publication-gates.sh --distributed --brew-caix "$(command -v caix)"
```

It must print `distributed is ready for Thunderbolt testing`. If it prints `not ready`, keep work
local.

## Stage Manifest Format

Use top-level `stages` in a standalone manifest. A copyable example lives at
[`docs/examples/cluster-stage-manifest.json`](examples/cluster-stage-manifest.json).

```json
{
  "schema": "caix.cluster.stage_manifest.v0",
  "model": "qwen3-0.6b-coreai",
  "total_layer_count": 28,
  "position_mode": "full_prefix",
  "boundary": {
    "hidden_state": {
      "name": "hidden_states",
      "shape": [1, -1, 1024],
      "scalar_type": "float16"
    }
  },
  "stages": [
    {
      "id": "embed",
      "role": "embeddings",
      "layers": "embeddings",
      "bundle": "stages/00-embed.aimodel",
      "memory_gb": 1.0
    },
    {
      "id": "layers-00-14",
      "role": "transformer_layers",
      "layers": [0, 14],
      "bundle": "stages/01-layers-00-14.aimodel",
      "memory_gb": 2.0
    },
    {
      "id": "layers-14-28",
      "role": "transformer_layers",
      "layers": [14, 28],
      "bundle": "stages/02-layers-14-28.aimodel",
      "memory_gb": 2.0
    },
    {
      "id": "head",
      "role": "final_norm_head",
      "layers": "norm+lm_head",
      "bundle": "stages/03-head.aimodel",
      "memory_gb": 1.0
    }
  ]
}
```

Or put the same `stages` array under `cluster` in a bundle `metadata.json`:

```json
{
  "metadata_version": "0.2",
  "kind": "llm",
  "name": "qwen3-0.6b-coreai",
  "assets": {"main": "model.aimodel"},
  "cluster": {
    "schema": "caix.cluster.stage_manifest.v0",
    "total_layer_count": 28,
    "stages": []
  }
}
```

Each manifest needs `model` and stage rows with `id`, `role`, `layers`, `bundle`, and `memory_gb`.
Set `total_layer_count` and `position_mode` explicitly for runtime handoff. `position_mode` is
`full_prefix` or `current`, matching the staged export. The dry-run planner can derive the layer
count from the last transformer layer range and will warn when it does. Use the runtime role names
`embeddings`, `transformer_layers`, and `final_norm_head`. For `transformer_layers`, `layers` is a
half-open `[lower, upper]` range. Bundle paths are resolved relative to the manifest file, or
relative to the model bundle when using `--model`.

For real staged exports, include `boundary.hidden_state`. `shape` is `[batch, sequence, hidden]`;
use `-1` for dynamic sequence length. `scalar_type` must be `float16` or `float32`.

## Current TODOs

- `caix cluster join --help` exists; worker runtime is not implemented.
- `caix serve --cluster` is advertised; coordinator runtime is not implemented.
- Stage export metadata is not emitted by the converter yet.
- Core AI stage execution exists for `.none` and `.stateful` stage graphs; `.explicitOutputs`
  remains fail-closed.
- Runtime tensor transport and worker protocols are intentionally outside this dry-run command.
