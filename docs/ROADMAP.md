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
- Fail-closed `caix cluster join` and `caix serve --cluster` CLI stubs.
- `DistributedSameMachinePipeline`, an in-process stage-handle harness tested with fake stages,
  manifest-ordered handle maps, and a stage-handle factory context with resolved stage asset paths.

Still missing before real staged inference:

- Stage exporter output for per-stage `.aimodel` bundles and `cluster.stages` metadata.
- A Core AI `DistributedStageHandle`.
- Token-for-token Qwen3-0.6B same-machine evidence against the monolithic bundle.
- Loopback worker/coordinator transport.
- Thunderbolt Bridge test evidence.

Why this path:

- Core AI exposes local `AIModel` / `InferenceFunction` execution with caller-owned input buffers,
  output views, and mutable KV state.
- The current public Core AI language runtime is local-device only.
- MLX/exo prove multi-Mac inference is practical, but caix needs a Core AI-native path for
  converted `.aimodel` bundles.

Do first:

1. Build the stage exporter and Core AI `DistributedStageHandle`.
2. Prove same-machine staged execution with Qwen3-0.6B.
3. Verify staged output matches the monolithic Core AI bundle.
4. Split the stages into two local processes over loopback.
5. Move one stage to the 32 GB MacBook over Thunderbolt Bridge.
6. Add the 16 GB Mac mini as a third shard.

MacBook test gate:

- Do not ask for Thunderbolt testing until steps 1-3 pass.
- First external test is one MacBook stage over Thunderbolt Bridge.
- Use the same staged Qwen3-0.6B manifest that passed loopback.
- `scripts/check-distributed-readiness.sh --brew-caix "$(command -v caix)"` must pass first.
- Test the release path through Brew before connecting the MacBook.

Do not start with:

- Tensor parallelism.
- All-reduce collectives.
- Shipping logits between machines.
- Claims that Mac memory is pooled.

The claim, once working: stage-sharded Core AI execution across Macs.
