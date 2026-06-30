# Cluster Planning

This page documents dry-run planning and the minimal staged-worker POC.

`caix cluster plan` reads local staged-bundle metadata and prints a placement plan. It does not
load Core AI models, start workers, transfer tensors, benchmark, download, or upload anything.
JSON output includes a `runtime_plan` object using the same `DistributedStagePlan` contract that
the runtime validates, including `boundary_tensor`.

`caix serve --cluster` and `caix cluster join` load Core AI stage bundles and run the staged POC.
Use tiny random staged assets for fast transport checks. Real Qwen remains gated on stage parity.

## Commands

```bash
caix cluster plan --manifest qwen3-stages.json --workers main=64,mbp=32,mini=16
caix cluster plan --model models/exports/qwen3-staged --worker main=64 --worker mini=16
caix cluster plan --manifest qwen3-stages.json --json
caix deploy verify --endpoint main.local:1237 --endpoint mbp.local:1237
caix serve --cluster stage-manifest.json --host 0.0.0.0 --prompt-tokens 1,2,3 --max-tokens 1 --join-timeout 120 --once
caix cluster join --coordinator main.local:1237 --manifest stage-manifest.json --stage stages/01-layers.aimodel --connect-timeout 120
```

`caix deploy verify` checks caix HTTP visibility and link speed across distinct machine identities.
It does not load models, run staged inference, or prove tensor transport.

`--model` reads `metadata.json` from the bundle and expects a `cluster.stages` block. Current
single-bundle exports do not include that block, so the command reports a TODO telling the exporter
what metadata is missing.

The planner preserves stage order and assigns each stage to the first worker, in the order supplied,
with enough remaining memory. Worker memory is a dry-run budget in GB.

## Thunderbolt Test Gate

Before asking for MacBook Thunderbolt testing, the gate is:

1. Same-machine staged POC works over loopback.
2. `caix deploy verify` sees both machines and reports acceptable link speed.
3. Tiny staged smoke runs through Brew-installed `caix` with explicit timeouts.
4. Real Qwen3-0.6B stays unpublished until parity/load gates pass.

Install caix through Brew on the test machine and run the installed check scripts, not checkout
binaries.

Use the tiny POC readiness gate before asking for hardware:

```bash
scripts/check-distributed-readiness.sh --tiny-poc \
  --tiny-manifest /path/to/qwen3-tiny-random-coreai-staged-rope-input-f16-2x1/stage-manifest.json \
  --brew-caix "$(command -v caix)"
```

It must print `tiny distributed POC is ready for Thunderbolt testing`. If it prints `not ready`,
keep work local. The real-Qwen publication gate remains `scripts/check-publication-gates.sh
--distributed --brew-caix "$(command -v caix)"`.

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

If staged `.aimodel` assets already exist under a bundle, attach their manifest to
`metadata.json` with:

```bash
python3 python/converter/convert.py --bundle <bundle> --attach-cluster-manifest <manifest.json>
```

This validates stage asset paths, `main.mlirb`, layer coverage, function maps, boundary metadata,
and final-stage `vocab_size`. It does not create staged assets.

## Current Status

- `caix cluster plan` validates manifests without loading models.
- `caix serve --cluster` and `caix cluster join` run the minimal socket-backed staged POC.
- `scripts/check-tiny-cluster-smoke.sh` prints or runs tiny staged smoke commands with timeouts.
- The converter can attach validated cluster metadata for existing staged assets, but staged asset
  creation remains a separate export path.
- Real Qwen3-0.6B staged bundles are not publishable until parity/load gates pass.
