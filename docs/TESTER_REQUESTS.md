# Tester Requests

Generated from `benchmarks/MANIFEST.tsv`.
Revision source: `benchmarks/revisions.tsv`.
Raw evidence source: `benchmarks/raw`.

No speed claims without raw logs. Use the exact revision in the table. Keep prompts, token budget,
temperature, seed, streaming mode, warmup count, measured run count, and chat-template mode unchanged.
Speculative and EAGLE/MTP rows are target+draft package evidence, not standalone target evidence.
Report target-only and target+draft results separately, and do not average or substitute one for the
other.

## Ready Benchmark Requests

| repo | revision | local dir | request | notes |
|---|---|---|---|---|
| `redhillsmediafl/rhm-qwen2.5-0.5b-instruct-bf16-caix` | `ec5aab1ef697058c3465a07ca8c3fc6e45ff083c` | `qwen2.5-0.5b-instruct-bf16-coreai-ctx32768` | load, generation, benchmark | full-native-context regular bf16; long-context/determinism/HF first-token gates passed; speed pending |
| `redhillsmediafl/rhm-qwen2.5-3b-instruct-bf16-caix` | `86e808e92a400a0debd01e3dfffd84aae6c11670` | `qwen2.5-3b-instruct-bf16-coreai-ctx32768` | load, generation, benchmark | full-native-context regular bf16; long-context/determinism/HF first-token gates passed; Qwen Research License; speed pending |
| `redhillsmediafl/rhm-qwen3-0.6b-bf16-caix` | `a37d147867643cb827086e42518baa254f55a43d` | `qwen3-0.6b-bf16-coreai-ctx40960` | load, generation, benchmark | full-native-context regular bf16; long-context/determinism/HF first-token gates passed; speed pending |
| `redhillsmediafl/rhm-qwen3-1.7b-bf16-caix` | `47e7570d368dc6f1f25c4c834bb990b7154f7d81` | `qwen3-1.7b-bf16-coreai-ctx40960` | load, generation, benchmark | full-native-context regular bf16 pattern-setter; long-context/determinism/HF first-token gates passed; speed pending |
| `redhillsmediafl/rhm-qwen3-8b-bf16-caix` | `802db482ce392f027238e04d27032f0192aaed34` | `qwen3-8b-bf16-coreai-ctx40960` | load, generation, benchmark | full-native-context regular bf16; long-context/determinism/HF first-token gates passed; speed pending |
| `redhillsmediafl/rhm-qwen3-14b-bf16-caix` | `96ec804b7d25c14826ae917ddb9bf6990d66163c` | `qwen3-14b-bf16-coreai-ctx40960` | load, generation, benchmark | full-native-context regular bf16; long-context/determinism/HF first-token gates passed; peak RSS ~48.6 GiB on Studio; speed pending |
| `redhillsmediafl/rhm-glm-4-9b-0414-bf16-caix` | `ffe18144a446993cb957973b4436ed6e96ef2eea` | `glm-4-9b-0414-bf16-coreai-ctx32768` | load, generation, benchmark | full-native-context regular bf16; long-context/determinism/HF first-token gates passed; speed pending |
| `redhillsmediafl/rhm-glm-4-32b-0414-int8-caix` | `ecd9e6f99d9cb61848fb86f0a58a681fce2d8292` | `glm-4-32b-0414-int8-coreai-ctx32768` | load, generation, benchmark | full-native-context regular int8; 17022-token serve, fresh-process determinism, and HF first-token gates passed; bf16/fp16 regular is outside the current 64GB conversion envelope; speed pending |
| `redhillsmediafl/rhm-gpt-oss-20b-int8-caix` | `6777e93b7dcd1739891330c70868834379d426fb` | `gpt-oss-20b-int8-coreai-ctx131072` | load, generation, benchmark | full-native-context regular int8; 17032-token serve, fresh-process determinism, and HF first-token gates passed; bf16/fp16 regular is outside the current 64GB conversion envelope; speed pending |
| `redhillsmediafl/rhm-mistral-7b-instruct-v0.3-bf16-caix` | `82d99d88875d6c89f9af5c906eb1b5cc66ed19fe` | `mistral-7b-instruct-v0.3-bf16-coreai-ctx32768` | load, generation, benchmark | full-native-context regular bf16; long-context/determinism/HF first-token gates passed; speed pending |
| `redhillsmediafl/rhm-mistral-nemo-instruct-2407-bf16-caix` | `c92d81f632e4f23b5e454a49a0579b4198afef6f` | `mistral-nemo-instruct-2407-bf16-coreai-ctx131072` | load, generation, benchmark | full-native-context regular bf16; long-context/determinism/HF first-token gates passed; peak RSS ~50 GiB on Studio; speed pending |
| `redhillsmediafl/rhm-mistral-small-instruct-2409-int8-caix` | `c779e46d7a84d34186d5c509545dc036981d6ddf` | `mistral-small-instruct-2409-int8-coreai-ctx32768` | load, generation, benchmark | full-native-context regular int8; 17038-token serve, fresh-process determinism, and HF first-token gates passed; bf16/fp16 regular is outside the current 64GB conversion envelope; Mistral Research License; speed pending |

## Existing Raw Evidence

| repo | revision | local dir | mode | measured runs | raw dir |
|---|---|---|---|---|---|
| `redhillsmediafl/rhm-qwen2.5-0.5b-instruct-caix` | `457592be4e87468a6c64f5567dc3bd46554daa13` | `qwen2.5-0.5b-instruct-coreai` | `decode` | 3 | `benchmarks/raw/20260628-214941-qwen2.5-0.5b-instruct-coreai` |
| `redhillsmediafl/rhm-qwen2.5-3b-instruct-caix` | `e0c019d5a534ec8aea936dfafedb7e00b17c3961` | `qwen2.5-3b-instruct-coreai` | `decode` | 3 | `benchmarks/raw/20260628-220044-qwen2.5-3b-instruct-coreai` |
| `redhillsmediafl/rhm-qwen3-0.6b-caix` | `3aa798b1a942fd15be6e5a96bd18b01e52dc6bc4` | `qwen3-0.6b-coreai` | `decode` | 3 | `benchmarks/raw/20260628-214406-qwen3-0.6b-coreai` |
| `redhillsmediafl/rhm-qwen3-1.7b-caix` | `bf04650aef9a5e325f1ba05c52866b0363f56c29` | `qwen3-1.7b-coreai` | `decode` | 3 | `benchmarks/raw/20260629-072107-qwen3-1.7b-coreai` |
| `redhillsmediafl/rhm-qwen3-8b-caix` | `e46668067aff8efb89e87469269c120073511136` | `qwen3-8b-coreai` | `decode` | 3 | `benchmarks/raw/20260628-222458-qwen3-8b-coreai` |
| `redhillsmediafl/rhm-qwen3-14b-caix` | `62a1e52fd5a5b9c1b241243cb0903d80fb5ad624` | `qwen3-14b-coreai` | `decode` | 3 | `benchmarks/raw/20260628-223516-qwen3-14b-coreai` |
| `redhillsmediafl/rhm-glm-4-9b-0414-caix` | `c759b1583693b3f051e62fd7082e4bc538ebb72c` | `glm-4-9b-0414-coreai` | `decode` | 3 | `benchmarks/raw/20260628-230705-glm-4-9b-0414-coreai` |
| `redhillsmediafl/rhm-glm-4-32b-0414-caix` | `59bddf6e8d498fd991144c4d47ab3b259e9a9d0b` | `glm-4-32b-0414-coreai` | `decode` | 3 | `benchmarks/raw/20260629-030527-glm-4-32b-0414-coreai` |
| `redhillsmediafl/rhm-gpt-oss-20b-caix` | `ae08b1c0dc03b6ddc53901adaa29e97d23b1cfdb` | `gpt-oss-20b-coreai` | `decode` | 3 | `benchmarks/raw/20260629-002744-gpt-oss-20b-coreai` |
| `redhillsmediafl/rhm-mistral-7b-instruct-v0.3-caix` | `2014f4967181dcab32c75ffb89dbd714f8f89910` | `mistral-7b-instruct-v0.3-coreai` | `decode` | 3 | `benchmarks/raw/20260628-233234-mistral-7b-instruct-v0.3-coreai` |
| `redhillsmediafl/rhm-mistral-nemo-instruct-2407-caix` | `9821a2de1f1029c0352b0d6311c6d399e9aac1fe` | `mistral-nemo-instruct-2407-coreai` | `decode` | 3 | `benchmarks/raw/20260628-234824-mistral-nemo-instruct-2407-coreai` |
| `redhillsmediafl/rhm-mistral-small-instruct-2409-caix` | `0549d42b45a65454bc2f99843deaebeab1587bb1` | `mistral-small-instruct-2409-coreai` | `decode` | 3 | `benchmarks/raw/20260629-000628-mistral-small-instruct-2409-coreai` |
| `redhillsmediafl/rhm-mixtral-8x7b-instruct-caix` | `ea180189c4266d8a0dde4e3238cf959789c0504f` | `mixtral-8x7b-instruct-coreai` | `decode` | 3 | `benchmarks/raw/20260629-042411-mixtral-8x7b-instruct-coreai` |

## Manual Or Component Requests

The staged distributed rows below need package-specific installed-caix hardware evidence before any
distributed readiness or speed claim. A generic two-machine POC, Studio-only loopback, plan dry-run,
or HF diagnostic parity does not substitute for the listed package's own distributed smoke.

| repo | revision | local dir | request | notes |
|---|---|---|---|---|
| `redhillsmediafl/rhm-gemma-4-12b-it-mm-staged-caix` | `f66fc8a975baf0c326af3595ca12004a0eb71181` | `gemma4-12b-it-mm-splitcache-masked-singleasset-staged-4bit-ctx256k-2x24` | distributed hardware smoke | staged multimodal split-cache package; Studio HTTP hardware smoke passed with caix 0.2.13-beta; no public speed row |
| `redhillsmediafl/rhm-gemma-4-e2b-it-mm-staged-caix` | `1c0f1e3e3d03d6e41b484ea3d859160ef25a3cf4` | `gemma4-e2b-it-mm-QAT-splitcache-singleasset-staged-4bit-ctx128k-2x13-22` | distributed hardware smoke | staged multimodal split-cache package; Studio HTTP hardware smoke passed with caix 0.2.13-beta; no public speed row |
| `redhillsmediafl/rhm-gemma-4-e4b-it-mm-staged-caix` | `538ee06ee1039073d0c0ef290d60d48dd3941136` | `gemma4-e4b-it-mm-QAT-splitcache-plefix-singleasset-staged-4bit-ctx128k-2x21` | distributed hardware smoke | staged multimodal split-cache package; 32 GB MacBook HTTP hardware smoke passed with caix 0.2.13-beta; no public speed row |
| `redhillsmediafl/rhm-gemma-4-26b-a4b-it-mm-staged-caix` | `978768211c44a170856e738d2776be5d636353c1` | `gemma4-26b-a4b-it-mm-splitcache-masked-singleasset-staged-4bit-ctx256k-5x6` | distributed hardware smoke | staged multimodal split-cache package; Studio HTTP hardware smoke passed with caix 0.2.13-beta; no public speed row |
| `redhillsmediafl/rhm-gemma-4-31b-it-mm-staged-caix` | `01ae74db2f9775bf0833dc83972ad4b892ec1c9b` | `gemma4-31b-it-mm-splitcache-masked-singleasset-staged-4bit-ctx256k-6x10` | distributed hardware smoke | staged multimodal split-cache package; Studio HTTP hardware smoke passed with caix 0.2.13-beta; full-native 262k requires 64 GB+ or multi-device; no public speed row |

## Run Template

Set one row's values:

```bash
REPO=<repo-from-table>
REVISION=<revision-from-table>
NAME=<local-dir-from-table>
```

Install one payload:

```bash
export HF_HOME=${HF_HOME:-/Volumes/SSD/hf-cache}
scripts/check-disk-pressure.sh --path /Volumes/SSD --floor-gib 500
mkdir -p models/exports
hf download "$REPO" \
  --revision "$REVISION" \
  --local-dir "models/exports/$NAME"
```

Verify:

```bash
caix_bin=${caix_bin:-.build/release/caix}
MODEL="models/exports/$NAME"

"$caix_bin" inspect --model "$MODEL"
"$caix_bin" run \
  --model "$MODEL" \
  --prompt "Name one primary color." \
  --max-tokens 32 \
  --temperature 0 \
  --verbose
```

Benchmark:

```bash
scripts/benchmark-model.sh \
  --model "models/exports/$NAME" \
  --name "$NAME" \
  --repo "$REPO" \
  --repo-revision "$REVISION" \
  --prompt "Write one factual sentence about local inference on Apple silicon." \
  --max-tokens 128 \
  --temperature 0 \
  --warmup 1 \
  --runs 3
```

For classic speculative rows, add the draft bundle:

```bash
scripts/benchmark-model.sh \
  --model "models/exports/$NAME" \
  --draft "models/exports/$NAME/draft" \
  --name "$NAME" \
  --repo "$REPO" \
  --repo-revision "$REVISION" \
  --prompt "Write one factual sentence about local inference on Apple silicon." \
  --max-tokens 128 \
  --temperature 0 \
  --warmup 1 \
  --runs 3
```

For EAGLE MTP rows, benchmark the package:

```bash
scripts/benchmark-eagle.sh \
  --package "models/exports/$NAME" \
  --name "$NAME" \
  --repo "$REPO" \
  --repo-revision "$REVISION" \
  --prompt "Write one factual sentence about local inference on Apple silicon." \
  --max-tokens 128 \
  --warmup 1 \
  --runs 3
```

Report the fields in `docs/TESTING.md`. Send the raw benchmark directory. Preview cleanup, then
remove only the payload you installed:

```bash
scripts/remove-export.sh --dry-run "$NAME"
scripts/remove-export.sh "$NAME"
scripts/check-disk-pressure.sh --path /Volumes/SSD --floor-gib 500
```
