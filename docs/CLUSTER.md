# Cluster Planning

Planned feature. This page documents the first user-visible dry-run surface only.

`caix cluster plan` reads local staged-bundle metadata and prints a placement plan. It does not
load Core AI models, start workers, transfer tensors, benchmark, download, or upload anything.

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

## Stage Manifest Format

Use top-level `stages` in a standalone manifest. A copyable example lives at
[`docs/examples/cluster-stage-manifest.json`](examples/cluster-stage-manifest.json).

```json
{
  "schema": "caix.cluster.stage_manifest.v0",
  "model": "qwen3-0.6b-coreai",
  "stages": [
    {
      "id": "embed",
      "role": "embeddings",
      "layers": "embeddings",
      "bundle": "stages/00-embed.aimodel",
      "memory_gb": 1.5
    },
    {
      "id": "layers-00-13",
      "role": "transformer_layers",
      "layers": [0, 13],
      "bundle": "stages/01-layers-00-13.aimodel",
      "memory_gb": 5.0
    },
    {
      "id": "head",
      "role": "final_norm_head",
      "layers": "norm+lm_head",
      "bundle": "stages/02-head.aimodel",
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
    "stages": []
  }
}
```

Each stage currently needs `id`, `role`, `layers`, `bundle`, and `memory_gb`. Use the runtime role
names `embeddings`, `transformer_layers`, and `final_norm_head`. For `transformer_layers`,
`layers` is a half-open `[lower, upper]` range. Bundle paths are resolved relative to the manifest
file, or relative to the model bundle when using `--model`.

## Current TODOs

- `caix cluster join` is not implemented.
- `caix serve --cluster` is not implemented.
- Stage export metadata is not emitted by the converter yet.
- Runtime tensor transport and worker protocols are intentionally outside this dry-run command.
