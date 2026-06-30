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

Still missing before real Qwen release:

- Token-for-token Qwen3-0.6B same-machine evidence against the monolithic bundle.
- Thunderbolt Bridge test evidence.

Why this path:

- Core AI exposes local `AIModel` / `InferenceFunction` execution with caller-owned input buffers,
  output views, and mutable KV state.
- The current public Core AI language runtime is local-device only.
- MLX/exo prove multi-Mac inference is practical, but caix needs a Core AI-native path for
  converted `.aimodel` bundles.

Do first:

1. Prove the tiny random Qwen3 staged POC over two machines through Brew.
2. Prove same-machine staged execution with Qwen3-0.6B.
3. Verify staged output matches the monolithic Core AI bundle.
4. Move one real Qwen stage to the 32 GB MacBook over Thunderbolt Bridge.
5. Add the 16 GB Mac mini as a third shard.

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
