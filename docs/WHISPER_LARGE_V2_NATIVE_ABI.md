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

It does not convert the full checkpoint. The real-source test reads the safetensors header
through `safe_open.get_slice`; it never materializes the 6.17 GB tensor payload. Every tokenizer,
normalizer, frontend, and generation asset required by the pinned snapshot is independently
locked in `models/whisper-large-v2-source.json`.

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
precomputes it once, then produces one token at a time while growing self K/V state. Its logits
match a monolithic causal decoder at every tested position, and instrumentation proves zero
cross-projection calls inside `decode_step`.

## Selected CoreAI ABI

The production contract is `caix.whisper-split.v1` with strategy
`explicit_cross_kv_bridge`:

1. `encode`
2. `load_cross_kv` exactly once per utterance/window
3. `decode_step` once per generated token

The encoder emits cross K/V as normal outputs. The host passes those tensors once to the
decoder's `load_cross_kv` entrypoint. All following token steps read that decoder-owned cross
state and mutate only self K/V (plus an identity mutation required by `coreai-torch` to classify
cross K/V as state).

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

## CoreAI state-sharing evidence

Installed authoring stack during this proof:

- `coreai-core==1.0.0b1`
- `coreai-torch==0.4.0`
- `torch==2.9.0`

`torch.export` plus the CoreAI decomposition pass proves the intended mutation signatures:

- `encode`: `cross_state`
- `decode_step`: `cross_state`, `self_state`

A cross tensor that is only read cannot be named in `state_names`; `coreai-torch` rejects it with
`Graph has 1 stateful inputs ... but state_names has 2 entries`. The identity mutation in
`decode_step` is therefore part of the ABI, not an optimization accident.

Native compilation is blocked by the installed beta compiler. The subprocess-isolated two-
entrypoint proof reaches CoreAI conversion, then aborts during versioned-IR conversion with
`expected AICode versioned location` and `Failed to convert to versioned IR`. Apple's own
single-entrypoint stateful control,
`TestKVCache::test_coreai` from the installed `coreai-models` checkout, also exits with SIGABRT on
the same stack. That control rules out CAIX's multi-entrypoint design as the cause.

The gated test records this exact blocker as xfail and removes job-scoped `.aimodel` directories
from the parent process even when the compiler aborts. Once Apple updates the compiler stack, the
same test becomes a normal runtime assertion automatically; unrelated failures remain failures.

## Verification commands

```bash
caix_whisper_root=/Volumes/SSD/caix/.worktrees/whisper-large-v2-author

CAIX_WHISPER_LARGE_V2_SNAPSHOT=/Volumes/SSD/hf-cache/models--openai--whisper-large-v2/snapshots/ae4642769ce2ad8fc292556ccea8e901f1530655 \
PYTHONPATH="$caix_whisper_root/python" \
uv run --directory /Volumes/SSD/ai-dev/coreai-gemma4/vendor/coreai-models/python \
pytest -q "$caix_whisper_root/python/tests/test_whisper_large_v2_checkpoint.py"

PYTHONPATH="$caix_whisper_root/python" \
uv run --directory /Volumes/SSD/ai-dev/coreai-gemma4/vendor/coreai-models/python \
pytest -q "$caix_whisper_root/python/tests/test_whisper_split_reference.py" \
  "$caix_whisper_root/python/tests/test_whisper_native_abi.py"

CAIX_RUN_COREAI_STATE_PROOF=1 CAIX_COREAI_PROBE_TMP_ROOT=/Volumes/SSD/caix/.tmp \
PYTHONPATH="$caix_whisper_root/python" \
uv run --directory /Volumes/SSD/ai-dev/coreai-gemma4/vendor/coreai-models/python \
pytest -q "$caix_whisper_root/python/tests/test_coreai_whisper_state_probe.py"
```
