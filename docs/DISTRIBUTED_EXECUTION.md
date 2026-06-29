# Distributed Execution (v0 architecture)

Status: design plus in-process harness. A typed plan/validation layer, dry-run planner command,
and same-machine fake-stage coordinator already landed in the tree (see §0). Core AI stage
execution and cross-process transport are not implemented yet. This document defines the
execution contract those pieces are heading toward, and the exact first milestone.

Goal (from `docs/ROADMAP.md`): run a model that does not fit on one Mac by splitting Core AI
execution into stage shards across Macs. Pipeline parallelism only. No tensor parallelism, no
all-reduce, no shipping logits between machines, no claim that memory is pooled.

Scope split with `docs/CLUSTER.md`: that page documents the `caix cluster plan` dry-run
command (manifest format, placement). This page documents the runtime — the stage function
contract, the hidden-state transport, the coordinator/worker loop, KV ownership, failure
modes, and the equivalence milestone. The two must agree on vocabulary; where they describe
the same thing this page defers to `CLUSTER.md` and `DistributedRuntime.swift`.

This is blunt on purpose. Where the current code cannot do something, it says so.

---

## 0. What already exists in the tree

- Monolithic single-device runtime (`Sources/PipelineRuntime/LLMEngine.swift`,
  `BundleManifest.swift`) — the equivalence oracle. See §1.
- A typed plan + validation layer (`Sources/PipelineRuntime/DistributedRuntime.swift`):
  - `DistributedStageRole` = `embeddings` | `transformer_layers` | `final_norm_head`
    These are the canonical role names; this document uses them.
  - `DistributedLayerRange` half-open `lower_bound ..< upper_bound`.
  - `DistributedStageDescriptor` (id, role, layer_range, asset_name, worker_id) and
    `DistributedStagePlan` (model_name, total_layer_count, stages, workers).
  - `DistributedWorkerEndpoint` (id, host, port, labels).
  - `DistributedHiddenStatePacketMetadata` — the activation-packet metadata; see §4.
  - `DistributedRuntimeValidation` — enforces exactly one `embeddings` first,
    exactly one `final_norm_head` last, one-or-more `transformer_layers` between, contiguous
    gap-free layer coverage `0 ..< total_layer_count`, unique stage/worker ids, and
    adjacency-only packet routing (`destination == source + 1`).
  - `DistributedStageManifest` — the shared loader for top-level staged manifests and
    `metadata.json` cluster blocks, including hidden-state boundary tensor metadata.
  - `DistributedHiddenStatePacket`, `DistributedStageHandle`, and
    `DistributedSameMachinePipeline` — an in-process coordinator harness tested with fake
    stages. It can be built from a `DistributedStageManifest` plus stage-handle map or
    stage-handle factory. Factory calls receive normalized stage metadata, runtime descriptor,
    boundary tensor, and resolved asset URL. The pipeline validates stage order, packet routes,
    boundary tensor shape/dtype, payload byte counts, and final-token handoff without loading Core
    AI models.
  - `DistributedWorkerMessage` frames for hello/ack, allocation, forward, reset, free, and error.
    The `FORWARD` frame carries token ids or hidden-state metadata plus position ids; tensor bytes
    stay outside JSON. Worker frames validate through one runtime entry point before execution.
    `DistributedWorkerMessageCodec` writes and reads one JSON header line per frame.
    `DistributedWorkerWireFrame` pairs that header with optional tensor bytes and rejects payload
    sizes that do not match the header. Full wire-frame encoding is `JSON header line` followed by
    the raw tensor payload.
  - `DistributedWorkerWireFrameStreamDecoder` incrementally parses socket reads into complete
    worker wire frames without treating tensor bytes as JSON.
  - `DistributedWorkerFrameExecutor` dispatches validated request frames to one
    `DistributedStageHandle` and returns `forward_result` frames. It tracks request allocation,
    step order, processed-token position, KV capacity, reset, and free state before forwarding.
    It is transport-independent worker logic for the loopback and Thunderbolt paths.
  - `DistributedWorkerHandshakeCoordinator` accepts or rejects worker `HELLO` frames, prevents
    duplicate stage claims, and reports missing startup stages before execution.
  - `DistributedLoopbackWorkerTransport` runs worker handshakes and requests through the same
    frame encoder, stream decoder, executor, and response path in-process. It is the socket
    contract before there is a socket. Worker execution failures are encoded as `ERROR` frames.
  - `DistributedRemoteStageHandle` wraps a worker frame round trip so the coordinator can mix local
    and remote stages through the same `DistributedStageHandle` interface. Worker `ERROR` frames
    preserve their code/detail and must match the active request and remote stage before the
    coordinator reports them.
  - `DistributedStagePlan.integrityHash()` gives coordinator and workers the same SHA-256 plan
    identity for handshake rejection.
- A dry-run planner CLI (`Sources/PipelineCLI/Cluster.swift`, wired in `main.swift`):
  `caix cluster plan --manifest … | --model …` with greedy worker assignment. Manifest schema
  `caix.cluster.stage_manifest.v0` documented in `docs/CLUSTER.md`. JSON output includes a
  `runtime_plan` from the shared manifest loader, including the boundary tensor contract. It does
  not load Core AI models, start workers, or move tensors.

**Gap:** there is no Core AI stage execution path. Nothing produces a per-stage `.aimodel`,
nothing emits an intermediate hidden state from a graph or feeds one back in, and there is no
cross-process forward. The same-machine harness is ready for real `StageHandle`s once the
exporter and Core AI stage wrapper exist; the monolithic path stays as the oracle.

---

## 1. The monolithic runtime contract (the oracle)

A converted model today is one `.aimodel` package exposing `main` (and optional `decode`) that
maps token ids straight to logits, whole transformer inside one graph
(`LLMEngine.swift:8-40`):

- `main`: inputs `input_ids` (Int32 `[1,-1]`), `position_ids` (Int32 `[1,-1]`); output
  `logits` (`[1,-1,vocab]`); KV state `keyCache` / `valueCache` over a dynamic sequence dim.
- Two cache contracts (`LLMEngine.swift:384-396`): `.stateful` (KV is graph state, mutated in
  place) and `.explicitOutputs` (KV passed in and returned as outputs).
- Causal mask is internal to the graph. The host feeds `position_ids` whose **length** sizes
  the causal window. Default is the full prefix `0 ..< processedTokenCount+n`;
  `COREAI_POS_MODE=current` selects the length-matched window (`LLMEngine.swift:629-646`).
- KV is allocated once at a fixed capacity and reused (`LLMEngine.swift:546-566`). Hybrid
  `qwen3_5` models pack a recurrent state into a fixed KV prefix and carry a `minKVCapacity`
  floor (`BundleManifest.swift:293-337`).
- Prefill feeds all prompt tokens at `0..<n`; decode feeds one token at `processedTokenCount`;
  the next token is sampled from the **last row** of `logits` (`LLMEngine.swift:202-353`).
- Logits dtype is restricted to float16/float32 because the typed-view API has no scalar for
  bfloat16 (`LLMEngine.swift:31-39, 466-474, 700-712`). This restriction drives the boundary
  dtype decision in §4.

Two engines drive this: `LLMEngine` (our explicit loop — we own every tensor and the KV;
this is what we extend for staging) and `PipelinedLLM` (Apple's `EngineFactory` fast path,
which hides the KV and the forward boundary and therefore **cannot** be staged). Staging uses
the explicit engine only.

---

## 2. Stage shard model

A model is cut into ordered stages along the residual stream, **between** decoder layers — the
only clean hand-off point. Roles map 1:1 to `DistributedStageRole`:

- `embeddings`: token embedding + decoder layers `[0 .. a)`. Owns KV for those layers.
  Consumes `input_ids`; produces a hidden state. Exactly one, first.
- `transformer_layers`: decoder layers `[a .. b)`. Owns KV for those layers. Consumes a hidden
  state; produces a hidden state. One or more, in the middle, layer ranges contiguous and
  gap-free.
- `final_norm_head`: trailing decoder layers + final norm + LM head. Owns KV for its layers.
  Consumes a hidden state; produces `logits` (and, in v0, samples locally — §5.4). Exactly
  one, last.

`DistributedRuntimeValidation.validate(plan:)` already enforces the ordering and contiguous
`0 ..< total_layer_count` coverage. KV for a stage's layers lives only on the worker holding
that stage and never crosses the wire. Only the residual-stream hidden state crosses a
boundary.

### 2.1 Stage package layout

A staged model is a set of per-stage `.aimodel` bundles plus a stage manifest. The manifest is
`caix.cluster.stage_manifest.v0` (full schema and the `cluster`-block-in-`metadata.json`
variant are in `docs/CLUSTER.md`); it keeps `metadata_version` 0.2 and adds a `cluster` block
rather than introducing a new bundle kind. The runtime needs each stage entry to resolve to:
its role, its layer range (for `transformer_layers`), and its `.aimodel` path. A worked
example is `docs/examples/cluster-stage-manifest.json`: `total_layer_count` sits at the top
level; each `transformer_layers` stage gives `layers` as a half-open `[lower, upper]` array;
the `embeddings` and `final_norm_head` stages give `layers` as a label string (`"embeddings"`,
`"norm+lm_head"`). Real staged exports also include `boundary.hidden_state` with tensor name,
shape `[batch, sequence, hidden]`, and scalar type (`float16` or `float32`).

Note: the dry-run manifest carries `memory_gb` per stage for placement. Tied-embedding models
(Qwen3 family) duplicate the embedding/unembedding matrix across the `embeddings` and
`final_norm_head` stages; that duplication is real and must be reflected in those two stages'
`memory_gb`.

---

## 3. Stage function IO contract

Each stage exposes `main` (prefill) and optionally `decode`, mirroring the monolithic contract
layer for layer. Only the first input and last output differ by role.

| Role | inputs | output | KV state |
|---|---|---|---|
| `embeddings`     | `input_ids` Int32 `[1,-1]`, `position_ids` Int32 `[1,-1]` | `hidden_states` floatX `[1,-1,H]` | its layers |
| `transformer_layers` | `hidden_states` floatX `[1,-1,H]`, `position_ids` Int32 `[1,-1]` | `hidden_states` floatX `[1,-1,H]` | its layers |
| `final_norm_head`  | `hidden_states` floatX `[1,-1,H]`, `position_ids` Int32 `[1,-1]` | `logits` floatX `[1,-1,vocab]` | its layers |

Rules carried over unchanged from the monolithic contract:
- `position_ids` is fed to **every** stage, not only `embeddings`. RoPE and causal masking run
  inside each decoder layer; the mask window is the `position_ids` length. The coordinator
  computes `position_ids` exactly as `LLMEngine` does and sends the **identical** array to
  every stage on every step.
- All stages agree on position mode (full prefix vs current window). The coordinator honors
  the mode the shards were exported with; mixing modes silently corrupts RoPE.
- KV capacity is allocated once per request per stage, floored to that stage's KV floor —
  identical math to `LLMEngine.resolvedCapacity` / `allocateKVCache` (`LLMEngine.swift:546-566`).
- `H` (`hidden_size`) is fixed across all boundaries. The boundary tensor rank is always 3
  (`[1, n, H]`): `n` = prompt length on prefill, `1` on decode.

`StageHandle` is `LLMEngine` reduced to one stage: same KV allocation, same `runForward`, but
its first input is a hidden state for `transformer_layers`/`final_norm_head` instead of
`input_ids`, and its output is a hidden state instead of logits for
`embeddings`/`transformer_layers`. Not `Sendable`; driven by one task at a time, like
`LLMEngine` and `PersistentModel` (`PersistentModel.swift:5-13`).

---

## 4. Hidden-state tensor contract

This is the wire payload and the most load-bearing part of the design. Its metadata is already
typed as `DistributedHiddenStatePacketMetadata`; this section pins down what the bytes are.

- **Shape**: `[batch, sequence, hidden]` = `[1, n, H]`, row-major, contiguous. The packet's
  `shape[1]` must equal `position_range.count` (already validated, `:466-469`). Producers must
  materialize a contiguous buffer; Core AI views can be strided
  (`LLMEngine.lastRow`/`allRows` read stride-aware, `:725-766`), so the staged path compacts
  before sending.
- **dtype**: `DistributedTensorScalarType` (`scalar_type` field) is `float16` or `float32`.
  Boundary packets must use one of those two types. A stage may compute internally with another
  precision, but the exported boundary tensor must be readable by the host and serialized as
  float16 or float32. The host typed-view API cannot read bfloat16 today; keep it off the wire
  until that changes.
- **Endianness**: little-endian (all targets are arm64; assert, do not convert).
- **`byte_count`**: `prod(shape) * scalar_type.byteWidth`, validated against `shape` +
  `scalar_type` on every packet.
- **Routing**: `source_stage_id` → `destination_stage_id` must be adjacent
  (`destination == source + 1`), validated by `DistributedRuntimeValidation.validate(packet:in:)`.
- **`step_index`**: monotonic per request — 0 = prefill, then 1,2,… per decode step. Used to
  detect dropped/duplicated forwards (§6).

### 4.1 Bandwidth reality (Qwen3-0.6B, H=1024, float16 = 2 bytes)

- Per boundary, per decode step: `1*1024*2 = 2048` bytes each direction.
- Per boundary, prefill of `n` tokens: `2048 * n` bytes.
- A 3-stage split has 2 internal boundaries; per decode step the hidden state crosses both
  (~4 KB moved). Negligible on a Thunderbolt Bridge link.
- **Why logits never cross the wire:** full logits per decode step are
  `vocab * 2 = 151936 * 2 ≈ 297 KB` — ~145× a hidden-state boundary, every step. The
  `final_norm_head` stage samples locally and returns a token id (4 bytes). This is the
  roadmap's "do not ship logits between machines" line, made quantitative.

---

## 5. Coordinator / worker protocol

The coordinator is a single process on the main Mac. It owns: tokenizer, the decode loop,
`position_ids` computation, stop-token / context-limit logic, and (greedy) the stop decision.
It owns **no** KV — each worker owns its stage's KV. A worker holds one stage (v0; co-locating
stages on one worker is a later optimization) plus that stage's graph, KV cache, and
`processedTokenCount`.

### 5.1 Control messages

Control frames are JSON. Tensor-bearing frames carry a `DistributedHiddenStatePacketMetadata`
header followed by the raw payload (§4).

- `HELLO` (worker → coordinator): stage id/role/layer_range, hidden_size, boundary dtype,
  cache contract, plan integrity hash, free memory, compute unit.
- `HELLO_ACK`: accept or reject. Reject on plan/integrity mismatch, an already-claimed stage,
  or a missing stage.
- `ALLOC` {request_id, kv_capacity}: worker allocates KV (floored to its KV floor) and resets
  `processedTokenCount = 0`. Mirrors `allocateKVCache`.
- `FORWARD` — §5.2.
- `RESET` {request_id}: worker rolls KV back to 0 (cheap; only the position counter moves —
  `LLMEngine.rollbackKV`, `:612-615`).
- `FREE` {request_id}: worker drops the KV for an active request. Unknown request IDs are
  rejected before stage teardown runs.
- `ERROR` {code, detail}: terminal for the current request.

### 5.2 Prefill flow

1. Coordinator tokenizes → `input_ids[0..n)`; computes `position_ids` (full prefix `0..<n`).
2. `ALLOC` to every stage with `kv_capacity = min(n + maxTokens + 8, max_context_length)`
   (per-stage floored).
3. `FORWARD` to the `embeddings` stage: `step_index=0`, `input_ids` as the tensor (Int32,
   `[1,n]`), `position_ids` in the header. Stage runs its layers, advances KV by `n`, returns
   `hidden_states [1,n,H]`.
4. Coordinator relays that hidden state to the next stage with the **same** `position_ids`;
   that stage advances KV by `n`, returns its hidden state. Repeat through the last
   `transformer_layers` stage.
5. `final_norm_head` runs its layers + final norm + head, reads the **last row** of logits,
   computes argmax (v0 = greedy), returns `{ token_id }`.
6. Coordinator records `token_id` as the first generated token.

### 5.3 Decode flow

Loop until EOS / `maxTokens` / context limit (all coordinator-owned, identical to
`LLMEngine.generate`, `:276-324`):

1. Coordinator has last token `t`; computes `position_ids` for length `processedTokenCount+1`.
2. `FORWARD` to `embeddings`: `step_index=k`, `input_ids=[t]` (`[1,1]`), `position_ids` in
   header. Stage advances KV by 1, returns `hidden_states [1,1,H]`.
3. Relay the hidden state through the `transformer_layers` stages (each advances KV by 1), same
   `position_ids`.
4. `final_norm_head` argmaxes the single logits row, returns `{ token_id }`.
5. Coordinator detokenizes, checks stop tokens (`LLMEngine.stopTokenIds`, `:777-798`), emits
   the delta, loops.

### 5.4 Sampling location

v0 acceptance is **greedy** (`temperature == 0`), so argmax is deterministic and the
`final_norm_head` stage samples locally — exactly equivalent to the monolithic path and cheap
on the wire. Temperature sampling with a seed is deferred: it needs one RNG owner. When added,
the head stage returns a top-k logit slice (id+value pairs, not the full vocab) and the
coordinator samples, keeping the existing `Sampler` + `SeededGenerator` (`:801-812`) as the
single RNG. Full-vocab logits are never returned.

---

## 6. KV ownership and lifecycle

- KV for layers `[a..b)` exists only on the worker holding that stage. Allocated on `ALLOC`,
  advanced by `n` per `FORWARD`, **not reconstructable** elsewhere without replaying the token
  sequence through that stage.
- `processedTokenCount` is tracked independently per worker but is identical across workers for
  a request (same sequence, same number of forwards). The coordinator includes the expected
  count in `FORWARD`; a worker rejects the call if its own counter disagrees. `step_index`
  monotonicity (§4) gives a second, cheaper drift check.
- Capacity is fixed at `ALLOC`; exceeding `max_context_length` ends the request with
  `contextLimit`, decided by the coordinator before the next `FORWARD`.
- Hybrid `qwen3_5` recurrent-state stages cannot be rolled back correctly (the SSM state is not
  a positional KV; `LLMEngine.swift:606-615` already restricts rollback to standard attention).
  v0 targets a standard-attention model (Qwen3-0.6B), so this does not bite yet, but each
  stage's KV floor must still be honored.

---

## 7. Failure modes

| Failure | Detection | v0 response |
|---|---|---|
| Worker crash / disconnect mid-request | stream EOF / `FORWARD` timeout | Fail the request. KV is lost and non-reconstructable; no silent partial output. Recovery = re-prefill from token 0. |
| Plan / shard version skew | integrity hash mismatch at `HELLO` | Reject the worker; refuse to start the cluster. |
| Boundary dtype mismatch | packet `scalar_type` ≠ plan boundary dtype | `ERROR`, fail request. |
| Hidden-state shape mismatch | packet `shape` ≠ `[1,n,H]`, or `shape[1]` ≠ `position_range.count` | `ERROR` (already a validation error, `:466-469`). |
| Non-adjacent packet route | `destination` ≠ `source + 1` | `ERROR` (`packetRouteMismatch`, `:498-502`). |
| `processedTokenCount` / `step_index` divergence | worker counter ≠ coordinator expected | `ERROR`, fail request (dropped/duplicated forward). |
| KV capacity / context exhaustion | coordinator pre-checks before `FORWARD` | Stop with `contextLimit`. |
| Stage missing at startup | not all stages claimed | Refuse to serve. |
| Network stall (no progress) | per-`FORWARD` deadline | Fail the request; never retry a forward (would double-advance KV). |
| Unsupported boundary dtype | rejected at plan load (§4) | Refuse the plan until the exporter emits float16 or float32 boundary tensors. |

No automatic failover in v0. A lost worker fails the in-flight request; recovery is restart +
re-prefill. This is honest given KV locality; mid-pipeline KV migration is out of scope.

---

## 8. CLI surface

Built today (`Sources/PipelineCLI/Cluster.swift`, documented in `docs/CLUSTER.md`):
- `caix cluster plan --manifest … | --model …` — dry-run placement. Read-only; no model load,
  no workers, no tensors.

Target (not built):
- `caix cluster join --coordinator <host:port> --stage <dir>` — run a worker for one stage.
- `caix serve --cluster <manifest>` — run the coordinator; same HTTP API as `serve` today, but
  the model handle is a staged engine instead of a single `PersistentModel`.

Auth between coordinator and workers is out of scope for milestones 1–4 (loopback, then a
trusted point-to-point Thunderbolt Bridge / LAN link). Any later pre-shared cluster secret on
the control channel must follow `scripts/check-token-handling.sh`: no env-var secret reads, no
secret in argv.

---

## 9. First milestone: same-machine staged Qwen3-0.6B equivalence

The only milestone that matters right now. It isolates the **staging math** from transport:
everything runs in one process, hidden states pass through `DistributedHiddenStatePacket`s
in memory — no sockets.

### 9.1 Prerequisite (define, do not start)

A stage exporter that cuts the existing Qwen3-0.6B `.aimodel` into stage bundles per §2/§3,
plus the `cluster.stages` manifest metadata that the planner already expects but the converter
does not yet emit (`docs/CLUSTER.md` "Current TODOs"). Minimum useful split is 3 stages so the
test exercises all three IO shapes: `embeddings` (`input_ids`→hidden), `transformer_layers`
(hidden→hidden), `final_norm_head` (hidden→logits). A 2-stage split skips the pure hidden→hidden
boundary. The committed example `docs/examples/cluster-stage-manifest.json` is one such split
(a 4-stage layout: one `embeddings`, two `transformer_layers`, one `final_norm_head`).
Conversions, downloads, and exports are out of scope for this agent — this milestone
is blocked on that exporter and says so.

### 9.2 Runtime to build

`DistributedSameMachinePipeline` now validates ordered stage handles and stage-to-stage packet
handoff with fake stages. The remaining runtime work is a concrete Core AI `StageHandle` and a
thin `StagedEngine` wrapper:

- Load N handles from a stage manifest (each handle = `LLMEngine` specialized to one
  stage bundle, reusing `AIModel.specialize` with the persistent compile cache,
  `LLMEngine.swift:106-123`).
- `generate(promptTokens:options:)` reuses the monolithic loop (`LLMEngine.generate`) but
  replaces the single `step` with: `embeddings` forward → relay hidden packet → … →
  `final_norm_head` forward → sample. Coordinator-owned `position_ids`, stop logic, and capacity
  math are unchanged.
- Greedy only for this milestone.

### 9.3 Equivalence procedure

1. Fixed prompt set (e.g. 8 prompts spanning short / multi-turn / code), `temperature = 0`,
   `maxTokens = 128`, fixed `kvCapacity`.
2. Oracle: run the **monolithic** Qwen3-0.6B bundle through `LLMEngine` (not `PipelinedLLM` —
   the explicit engine is the reference) and record the generated token-id sequence per prompt.
3. Candidate: run the same prompts through `StagedEngine` over the 3-stage split.
4. Compare.

### 9.4 Acceptance criteria

- **Primary**: per-prompt generated token-id sequences are **identical** (token-for-token) for
  all 128 steps across the whole prompt set. Greedy argmax over the same residual stream is
  deterministic, so anything short of an exact match is a real bug, not numerical noise.
- **Diagnostic** (when primary fails, to localize drift before it flips a token): at step 0,
  compare the head stage's last-row logits against the monolithic last-row logits — they should
  match within float16 rounding. Also assert each internal boundary hidden state is finite
  (no NaN/Inf) and has shape `[1, n, H]`.
- **KV check**: after each forward, every stage's `processedTokenCount` equals the
  coordinator's expected count.

Passing 9.4 proves the cut points, the hidden-state contract, the per-stage KV, and the
position handling are correct **before** any byte crosses a process or machine boundary. Only
then do milestones 2+ (loopback processes → Thunderbolt Bridge → Mac mini third shard → CLI)
add transport, per §4/§5/§7.

### 9.5 What this milestone does not prove

- Nothing about throughput. A 3-stage split on one machine serializes three graph calls per
  token and shares one GPU; it will not be faster than monolithic and is not meant to be.
- Nothing about temperature sampling (greedy only).
- Nothing about multi-machine transport, serialization, or failure handling — those are
  milestones 2+ and depend on §4/§5/§7, designed here but unbuilt.

---

## 10. Open questions / contract risks

- **Boundary dtype export policy (§4).** `DistributedTensorScalarType` is float16/float32 only.
  Staged metadata can record the boundary tensor contract now; for internally bfloat16 graphs, the
  exporter still needs to materialize a host-readable boundary.
- **Staged exporter handoff.** The runtime and CLI can load staged manifests, but current exports
  still do not record staged metadata. The exporter must write `cluster.stages`,
  `total_layer_count`, stage asset names, and boundary tensor metadata before same-machine staged
  equivalence can run.
- **`AIModel.specialize` cache keying.** Does the `.default` compile cache key cleanly per
  stage graph, or do same-named functions (`main`/`decode`) across stage bundles collide?
  Settle during 9.2.
- **Qwen3-0.6B export specifics.** Confirm `.stateful` KV and float16 logits via the standard
  `coreai.llm.export` path (decides the default cache contract / boundary dtype), and the
  exported position mode (`full` vs `current`), which must be recorded in the manifest and fed
  identically to all stages.
- **Tied-embedding memory.** Per-stage memory after the embedding/unembedding duplication on
  the `embeddings` and `final_norm_head` stages — feeds `caix cluster plan` and the eventual
  placement on the 32 GB / 16 GB workers.
