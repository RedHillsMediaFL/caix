# Whisper large-v2 native authoring foundation

## Pinned source and authoring identity

This foundation converts exactly:

- repository: `openai/whisper-large-v2`
- revision: `ae4642769ce2ad8fc292556ccea8e901f1530655`
- weights: `model.safetensors`, 6,173,370,152 bytes
- weight SHA-256: `57a1ba2a82c093cabff2541409ae778c97145378b9ddfa722763cb1cb8f9020b`
- source precision: FP32
- native runtime precision: FP16

Source validation streams the 6.17 GB weight object through SHA-256 from an `O_NOFOLLOW`
regular-file descriptor. Safetensors consumes that same open descriptor. The converter constructs
the Transformers model on the meta device, assigns one FP16 partition at a time, and never
materializes an FP32 model. Every tokenizer, normalizer, frontend, generation, and configuration
file is independently locked by `models/whisper-large-v2-source.json`. Configuration JSON is read
through its authenticated descriptor, not reopened by pathname.

The authoring code is also immutable. `models/coreai-models-authoring-source.json` pins repository
commit `e666cdc9848fd17f41e43504bc574c8964812c9e` and package tree
`b2803957eee13084d06924cfc567a770379234ae`. Conversion archives that commit into task-scoped
temporary storage and imports only its extracted `python/src`; dirty or untracked files in the
working checkout cannot become authoring input. The exact package stack is:

- `coreai-core==1.0.0b2`
- `coreai-opt==0.2.0`
- `coreai-torch==0.4.1`
- `torch==2.9.0`
- `transformers==4.57.6`

## Weight split

The pinned checkpoint has exactly 1,259 FP32 tensors. CAIX rejects missing, unexpected,
mis-shaped, or mis-typed entries before conversion.

| Partition | Tensor count | Contents |
| --- | ---: | --- |
| Encoder/cross-KV | 583 | `model.encoder.*` plus every decoder cross-attention K and V projection |
| Decoder | 676 | Embeddings, decoder norms/MLPs, self-attention, and cross-attention Q/output projections |

`model.decoder.embed_tokens.weight` is both the input embedding and output projection. No
`proj_out.weight` exists at this revision, so the partition preserves object identity instead of
duplicating the 51,865 × 1,280 table.

Cross K/V projection occurs once in `encode`. `decode_step` performs no cross projection and uses
indexed writes into fixed `[layers, batch, heads, 448, head_dim]` self K/V buffers. The eager
reference matches a monolithic causal decoder, resets every state tensor, writes slot 447, and
rejects position 448 without mutation.

## Native ABI v2

The production contract is `caix.whisper-split.v2` with strategy
`explicit_cross_kv_bridge`:

1. `encode(input_features) -> cross_key_payload, cross_value_payload`
2. `load_cross_kv(payloads, state) -> load_status` exactly once
3. `decode_step(token_id, state) -> logits, decode_status` once per generated token

Both statuses are Int32 `[1]`. Exact value `[1]` means success; `[0]` means invalid state.

`load_cross_kv` succeeds only when `cross_ready == 0`. It uses gated additive initialization
because CoreAI b2 lowers that operation reliably, then transitions readiness to one. A second load
returns `[0]` and leaves every state tensor bit-identical.

`decode_step` succeeds only when `cross_ready == 1` and `0 <= position < 448`. Success writes the
current self K/V slot, increments position, returns finite logits, and returns `[1]`. Any other
readiness or position returns zeroed unusable logits and `[0]`; cross caches, self caches,
readiness, and position remain bit-identical. The eventual resident host must check the status and
bound position before accepting output.

| Tensor | Dtype | Shape | Role |
| --- | --- | --- | --- |
| `input_features` | Float16 | `[1, 80, 3000]` | 30-second log-mel window |
| `cross_key_payload` | Float16 | `[32, 1, 20, 1500, 64]` | Encoder-produced cross keys |
| `cross_value_payload` | Float16 | `[32, 1, 20, 1500, 64]` | Encoder-produced cross values |
| `cross_key_cache` | Float16 | `[32, 1, 20, 1500, 64]` | Decoder cross-key state |
| `cross_value_cache` | Float16 | `[32, 1, 20, 1500, 64]` | Decoder cross-value state |
| `self_key_cache` | Float16 | `[32, 1, 20, 448, 64]` | Decoder self-key state |
| `self_value_cache` | Float16 | `[32, 1, 20, 448, 64]` | Decoder self-value state |
| `token_id` | Int32 | `[1, 1]` | One decoder token |
| `position` | Int32 | `[1]` | Current decoder position |
| `cross_ready` | Int32 | `[1]` | Successful one-time load marker |
| `load_status` | Int32 | `[1]` | Loader success or invalid-state marker |
| `logits` | Float16 | `[1, 1, 51865]` | Next-token logits; zero on failure |
| `decode_status` | Int32 | `[1]` | Decoder success or invalid-state marker |

Cross state occupies 234.375 MiB and self state occupies 70 MiB, for 304.375 MiB of fixed state
per active decoder session. Request audio, features, decoder state, and transcripts remain
memory-only; the ABI permits no temporary audio files.

## Deliberate timestamp boundary

Only deterministic text decoding with Whisper's authenticated no-timestamps prefix is available.
Timestamp-token decoding, segment timestamps, word timestamps, and word/segment alignment are
intentionally unsupported in this correctness release. `includeTimestamps == true` throws the
typed `WhisperDecodingPolicy.PolicyError.timestampsUnsupported` before the first decoder call.
There is no partial timestamp implementation.

## Embedded provenance and pre-load authentication

Every accepted asset directory contains exactly four no-follow regular files:

- `metadata.json`
- `main.mlirb`
- `main.hash`
- `caix-manifest.json`

The canonical CAIX manifest pins the complete v2 functions, call order, tensors, states, source
weights, authoring stack, authoring-source commit/tree, and the size and SHA-256 of `main.mlirb`.
Validation streams `main.mlirb`, requires `main.hash` to be its exact raw 32-byte SHA-256, then
requires the manifest digest and every pinned field to match. Absent or extra entries, extra JSON
fields, malformed or non-canonical JSON, symlinks, and non-regular files fail closed.

Manifest validation completes before `AIModel.load`. The verifier then proves:

- exact encoder-payload equality after cross-cache load;
- success-status values and dtypes;
- nonzero current self K/V slots and zero unused tails;
- zero status/logits plus exact state equality for decode-before-load, repeated load, invalid
  readiness, negative position, and position 448;
- exact recorded peak RSS and asset-load, encode, cross-load, and decode latencies.

The tiny saved-CoreAI proof exercises the same v2 transitions with the real compiler/runtime and
also compares encoder payloads and decoder logits to the eager split model.

## Bounded export and recoverable publication

The export script uses a 12 GiB recorded-peak-RSS threshold. This is a check of the process high
water mark reported by `getrusage`, not an operating-system hard memory limit. The exporter checks
recorded peak RSS before graph save and again immediately before publication. It also requires
space for the expected 3.1 GB asset plus 2 GiB of headroom before model materialization.

CoreAI first saves into a unique sibling staging directory. CAIX writes and fsyncs the manifest,
renames to an absent sibling candidate, revalidates all four files, and only then publishes. A
missing destination uses an exclusive atomic rename. Replacing an authenticated v2 destination
uses APFS `RENAME_SWAP`, retaining the prior asset at the candidate path until the new final is
validated and the parent directory is fsynced. A failed post-swap check rolls back; if rollback or
old-asset cleanup fails, both authenticated assets are retained and a recovery-specific error is
reported. Task-created staging/candidate paths are removed after ordinary pre-publication failure.
An unauthenticated legacy destination is never overwritten.

For a production refresh, export to a distinct visible candidate path, run the authenticated full
verifier on that candidate, and only then promote it to the production final. The controller owns
the memory-release decision for the full conversion and load.

## Evidence status and commands

The eager tests and tiny saved-CoreAI v2 state-machine proofs pass. A prior v1 artifact measured
3,087,426,620 bytes, 9,616,343,040 bytes export peak RSS, 0.447207 seconds warm load, 0.289177
seconds encode, and 0.176574 seconds decode. Those historical measurements are sizing evidence
only: the v1 asset lacks statuses and `caix-manifest.json` and is not accepted as v2 evidence. Fresh
v2 full-export hash, size, peak RSS, and latencies remain controller-held until the explicit memory
release.

```bash
caix_whisper_root="$(git rev-parse --show-toplevel)"
coreai_python=/Volumes/SSD/caix/.tmp/coreai-b2-probe-20260722/bin/python
coreai_models_python=/Volumes/SSD/ai-dev/coreai-gemma4/vendor/coreai-models/python/src
candidate=/Volumes/SSD/caix-kept-exports/speech/whisper-large-v2-fp16-v2-candidate.aimodel

# Bounded pure/reference coverage.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$caix_whisper_root/python" \
  "$coreai_python" -m pytest -q \
  "$caix_whisper_root/python/tests/test_whisper_native_abi.py" \
  "$caix_whisper_root/python/tests/test_whisper_split_reference.py"

# Tiny saved-CoreAI state-machine proof; this does not load the full checkpoint.
CAIX_RUN_WHISPER_FULL_EXPORT_PROOF=1 \
CAIX_COREAI_PROBE_TMP_ROOT=/Volumes/SSD/caix/.tmp \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$caix_whisper_root/python:$coreai_models_python" \
  "$coreai_python" -m pytest -q \
  "$caix_whisper_root/python/tests/test_whisper_full_coreai_export.py"

# Controller-held full steps: release memory, export to a new visible candidate, then verify.
"$caix_whisper_root/scripts/export-whisper-large-v2-coreai.sh" "$candidate"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$caix_whisper_root/python" \
  "$coreai_python" -m whisper_large_v2.verify \
  --asset "$candidate" \
  --max-resident-gib 12
```
