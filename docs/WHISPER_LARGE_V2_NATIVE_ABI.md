# Whisper large-v2 native authoring foundation

## Status

This branch defines and tests the checkpoint contract, weight split, streaming decoder
reference, and CoreAI tensor ABI for exactly:

- repository: `openai/whisper-large-v2`
- revision: `ae4642769ce2ad8fc292556ccea8e901f1530655`
- weight object: `model.safetensors`, 6,173,370,152 bytes, SHA-256
  `57a1ba2a82c093cabff2541409ae778c97145378b9ddfa722763cb1cb8f9020b`
- source precision: FP32
- native runtime precision target: FP16

The full checkpoint is converted to one native FP16 CoreAI program with the three entrypoints
below. Source validation always streams all 6.17 GB through SHA-256 from an `O_NOFOLLOW`
regular-file descriptor, then reads safetensors tensors from that same still-open descriptor. The
converter constructs the Transformers model on the meta device and assigns one FP16 partition at a
time; it never materializes an FP32 model. Hugging Face snapshot symlinks are supported, but their
target filenames are never treated as identity. Every tokenizer, normalizer, frontend, and
generation asset required by the pinned snapshot is independently locked in
`models/whisper-large-v2-source.json` through the same descriptor-bound validation path.

## Weight consumption and split

The pinned checkpoint has exactly 1,259 FP32 tensors. CAIX rejects missing, unexpected,
mis-shaped, or mis-typed entries before conversion.

| Partition | Tensor count | Contents |
| --- | ---: | --- |
| Encoder/cross-KV | 583 | All `model.encoder.*` tensors plus every decoder layer's cross-attention K projection and V projection/bias |
| Decoder | 676 | Token/position embeddings, decoder norms/MLPs, self-attention, and cross-attention Q/output projections |

`model.decoder.embed_tokens.weight` is both the input embedding and output projection. No
`proj_out.weight` exists at this revision; the partition preserves object identity instead of
duplicating the 51,865 × 1,280 table.

Cross K/V projection is part of the encoder artifact. The tested two-layer PyTorch reference
precomputes it once, explicitly loads it once, then produces one token at a time using indexed
writes into fixed `[layers, batch, heads, 448, head_dim]` self K/V buffers. `position` and
`cross_ready` are mutable Int32 tensors, matching the native ABI. Its logits match a monolithic
causal decoder at every tested position, instrumentation proves zero cross-projection calls inside
`decode_step`, reset clears every state tensor, slot 447 is writable, and token 449 is rejected
without mutation.

## Selected CoreAI ABI

The production contract is `caix.whisper-split.v1` with strategy
`explicit_cross_kv_bridge`:

1. `encode`
2. `load_cross_kv` exactly once per utterance/window
3. `decode_step` once per generated token

The encoder emits cross K/V as normal outputs. The host allocates zeroed decoder state and passes
the payloads once to `load_cross_kv`. The native loader adds each payload into its zeroed cross
cache and increments `cross_ready` from zero to one. Additive loading is equivalent to copy under
the enforced reset/load-once contract and avoids a CoreAI b2 runtime-compiler crash on full-state
`copy_` or full-range slice replacement. All following token steps read decoder-owned cross state,
perform indexed mutable-slice writes into self K/V, and increment `position`. Identity mutations
keep read-only cross K/V and `cross_ready` addressable as CoreAI state.

| Tensor | Dtype | Shape | Role |
| --- | --- | --- | --- |
| `input_features` | Float16 | `[1, 80, 3000]` | 30-second Whisper log-mel window |
| `cross_key_cache` | Float16 | `[32, 1, 20, 1500, 64]` | Per-layer encoder-attention keys |
| `cross_value_cache` | Float16 | `[32, 1, 20, 1500, 64]` | Per-layer encoder-attention values |
| `self_key_cache` | Float16 | `[32, 1, 20, 448, 64]` | Stateful decoder keys |
| `self_value_cache` | Float16 | `[32, 1, 20, 448, 64]` | Stateful decoder values |
| `token_id` | Int32 | `[1, 1]` | One decoder token |
| `position` | Int32 | `[1]` | Current decoder position |
| `cross_ready` | Int32 | `[1]` | Successful one-time load marker |
| `logits` | Float16 | `[1, 1, 51865]` | Next-token logits |

The cross caches occupy 234.375 MiB together. The self caches occupy 70 MiB together. This
304.375 MiB state allocation is fixed per active decoder session and must remain memory-only.
CAIX must not persist features, audio, partial transcripts, final transcripts, or state tensors,
and this ABI does not permit temporary audio files.

## CoreAI three-entrypoint evidence

Successful authoring/runtime stack:

- `coreai-core==1.0.0b2`
- `coreai-torch==0.4.1`
- `coreai-opt==0.2.0`
- `torch==2.9.0`
- `transformers==4.57.6`

The executable proof converts one AIProgram containing the actual ABI entrypoints:

1. `encode(features) -> cross_key_payload, cross_value_payload`
2. `load_cross_kv(payloads, state) -> load_marker`
3. `decode_step(token, state) -> logits`

The graph-level mutation signatures are empty for `encode`; cross key, cross value, and
`cross_ready` for `load_cross_kv`; and cross key, cross value, self key, self value, `position`, and
`cross_ready` for `decode_step`. The native runtime proof calls the loader exactly once, calls the
decoder twice, verifies logits `24.5` then `28.5`, verifies cross state remains loaded, verifies
self K/V slots 0 and 1 contain the expected values while the remaining 446 slots are zero, and
verifies `position == 2` and `cross_ready == 1`.

The older `coreai-core==1.0.0b1` / `coreai-torch==0.4.0` stack aborts during versioned-IR
conversion. The gated test xfails only on that exact pair of compiler diagnostics; unrelated
conversion, compilation, and runtime failures remain failures. On b2/0.4.1, Apple's unchanged
`TestKVCache::test_coreai` control and CAIX's three-entrypoint proof both pass.

The test parent creates a unique probe directory and removes it in `finally`, so a compiler signal
cannot strand `.aimodel` contents. The child also uses a scoped temporary asset during normal
execution. Request audio, features, transcripts, and decoder state are never written by this ABI.

## Full artifact evidence

`scripts/export-whisper-large-v2-coreai.sh` pins the snapshot, source contract, Python interpreter,
authoring stack, and a 42 GiB hard resident-memory cap. The converter refuses an existing output,
saves through a sibling staging directory, atomically renames the completed asset, and removes the
staging directory on either success or failure.

The complete export on this 64 GiB machine produced:

- asset: `/Volumes/SSD/caix-kept-exports/speech/whisper-large-v2-fp16.aimodel`
- schema: `caix.whisper-split.v1`
- files: `metadata.json`, `main.mlirb`, and `main.hash`
- exact total size: 3,087,426,620 bytes
- exact export peak RSS: 9,616,343,040 bytes (8.956 GiB)
- exact standalone checkpoint-load peak RSS: 9,616,982,016 bytes (8.956 GiB)

The saved artifact was then loaded by CoreAI and executed through `encode`, `load_cross_kv`, and
`decode_step`. A cold gated pytest completed in 16.13 seconds. A direct warm verification measured
0.447207 seconds for asset load, 0.289177 seconds for the 30-second zero-feature encoder window,
and 0.176574 seconds for one decoder step, with peak RSS 10,224,779,264 bytes (9.522 GiB). It
verified exact cross-cache and logits shapes, finite logits, `cross_ready == 1`, `position == 1`,
and zeroes in every unused self-cache slot.

## Verification commands

```bash
caix_whisper_root="$(git rev-parse --show-toplevel)"
coreai_python=/Volumes/SSD/caix/.tmp/coreai-b2-probe-20260722/bin/python
coreai_models_python=/Volumes/SSD/ai-dev/coreai-gemma4/vendor/coreai-models/python/src
whisper_asset=/Volumes/SSD/caix-kept-exports/speech/whisper-large-v2-fp16.aimodel

CAIX_WHISPER_LARGE_V2_SNAPSHOT=/Volumes/SSD/hf-cache/models--openai--whisper-large-v2/snapshots/ae4642769ce2ad8fc292556ccea8e901f1530655 \
PYTHONPATH="$caix_whisper_root/python" \
uv run --directory /Volumes/SSD/ai-dev/coreai-gemma4/vendor/coreai-models/python \
pytest -q "$caix_whisper_root/python/tests/test_whisper_large_v2_checkpoint.py"

PYTHONPATH="$caix_whisper_root/python" \
uv run --directory /Volumes/SSD/ai-dev/coreai-gemma4/vendor/coreai-models/python \
pytest -q "$caix_whisper_root/python/tests/test_whisper_split_reference.py" \
  "$caix_whisper_root/python/tests/test_whisper_native_abi.py"

CAIX_RUN_COREAI_STATE_PROOF=1 CAIX_COREAI_PROBE_TMP_ROOT=/Volumes/SSD/caix/.tmp \
PYTHONPATH="$caix_whisper_root/python:$coreai_models_python" \
"$coreai_python" -m pytest -q \
  "$caix_whisper_root/python/tests/test_coreai_whisper_state_probe.py"

# Full conversion. The destination must not already exist.
"$caix_whisper_root/scripts/export-whisper-large-v2-coreai.sh" "$whisper_asset"

# Execute the complete saved artifact under the same memory cap.
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$caix_whisper_root/python:$coreai_models_python" \
"$coreai_python" -m whisper_large_v2.verify \
  --asset "$whisper_asset" \
  --max-resident-gib 42
```
