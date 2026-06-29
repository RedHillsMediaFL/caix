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

Why this path:

- Core AI exposes local `AIModel` / `InferenceFunction` execution with caller-owned input buffers,
  output views, and mutable KV state.
- The current public Core AI language runtime is local-device only.
- MLX/exo prove multi-Mac inference is practical, but CAIX needs a Core AI-native path for
  converted `.aimodel` bundles.

Do first:

1. Prove same-machine staged execution with Qwen3-0.6B.
2. Verify staged output matches the monolithic Core AI bundle.
3. Split the stages into two local processes over loopback.
4. Move one stage to the 32 GB MacBook over Thunderbolt Bridge.
5. Add the 16 GB Mac mini as a third shard.
6. Add `caix cluster plan`, `caix cluster join`, and `caix serve --cluster`.

Do not start with:

- Tensor parallelism.
- All-reduce collectives.
- Shipping logits between machines.
- Claims that Mac memory is pooled.

The claim, once working: stage-sharded Core AI execution across Macs.
