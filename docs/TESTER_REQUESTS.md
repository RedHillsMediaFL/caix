# Tester Requests

Generated from `benchmarks/MANIFEST.tsv`.
Revision source: `benchmarks/revisions.tsv`.
Raw evidence source: `benchmarks/raw`.

No speed claims without raw logs. Use the exact revision in the table. Keep prompts, token budget,
temperature, streaming mode, warmup count, measured run count, and chat-template mode unchanged.

## Ready Benchmark Requests

| repo | revision | local dir | request | notes |
|---|---|---|---|---|
| `redhillsmediafl/rhm-ornith-1.0-9b-caix` | `06656dec7e8165ff714729307ed75eabc3b8c1b5` | `ornith-1.0-9b-coreai` | load, generation, benchmark | cataloged |
| `redhillsmediafl/rhm-mixtral-8x7b-instruct-caix` | `ea180189c4266d8a0dde4e3238cf959789c0504f` | `mixtral-8x7b-instruct-coreai` | load, generation, benchmark | cataloged |
| `redhillsmediafl/rhm-gemma-4-26b-a4b-mtp-caix` | `98793a039e2a548f04668acb4c254f57ff16f145` | `gemma-4-26b-a4b-mtp-coreai` | EAGLE MTP load, generation, benchmark | benchmark MTP package; compare against standalone target row |
| `redhillsmediafl/rhm-gemma-4-31b-it-mtp-caix` | `0c68d17473cdc0fee45e4d27540d50897f81334a` | `gemma-4-31b-it-mtp-coreai` | EAGLE MTP load, generation, benchmark | benchmark MTP package; compare against standalone target row |
| `redhillsmediafl/rhm-qwen3-4b-mtp-caix` | `4190150f7a47b113d36ea679c4541d95a21ce3f6` | `qwen3-4b-mtp-coreai` | classic speculative load, generation, benchmark | benchmark classic target+draft package; compare against standalone target row |

## Existing Raw Evidence

| repo | revision | local dir | mode | measured runs | raw dir |
|---|---|---|---|---|---|
| `redhillsmediafl/rhm-qwen2.5-0.5b-instruct-caix` | `457592be4e87468a6c64f5567dc3bd46554daa13` | `qwen2.5-0.5b-instruct-coreai` | `decode` | 3 | `benchmarks/raw/20260628-214941-qwen2.5-0.5b-instruct-coreai` |
| `redhillsmediafl/rhm-qwen2.5-3b-instruct-caix` | `e0c019d5a534ec8aea936dfafedb7e00b17c3961` | `qwen2.5-3b-instruct-coreai` | `decode` | 3 | `benchmarks/raw/20260628-220044-qwen2.5-3b-instruct-coreai` |
| `redhillsmediafl/rhm-qwen3-0.6b-caix` | `3aa798b1a942fd15be6e5a96bd18b01e52dc6bc4` | `qwen3-0.6b-coreai` | `decode` | 3 | `benchmarks/raw/20260628-214406-qwen3-0.6b-coreai` |
| `redhillsmediafl/rhm-qwen3-4b-caix` | `bc5b7f1b866dee97270d4e1d7bdbdd30e48aa641` | `qwen3-4b-coreai` | `decode` | 3 | `benchmarks/raw/20260628-221149-qwen3-4b-coreai` |
| `redhillsmediafl/rhm-qwen3-8b-caix` | `e46668067aff8efb89e87469269c120073511136` | `qwen3-8b-coreai` | `decode` | 3 | `benchmarks/raw/20260628-222458-qwen3-8b-coreai` |
| `redhillsmediafl/rhm-qwen3-14b-caix` | `62a1e52fd5a5b9c1b241243cb0903d80fb5ad624` | `qwen3-14b-coreai` | `decode` | 3 | `benchmarks/raw/20260628-223516-qwen3-14b-coreai` |
| `redhillsmediafl/rhm-qwythos-9b-caix` | `d4783ebca55aa4bf52d9bb4b254266ac5260d9c2` | `qwythos-9b-coreai` | `decode` | 3 | `benchmarks/raw/20260628-224859-qwythos-9b-coreai` |
| `redhillsmediafl/rhm-glm-4-9b-0414-caix` | `c759b1583693b3f051e62fd7082e4bc538ebb72c` | `glm-4-9b-0414-coreai` | `decode` | 3 | `benchmarks/raw/20260628-230705-glm-4-9b-0414-coreai` |
| `redhillsmediafl/rhm-glm-4-32b-0414-caix` | `59bddf6e8d498fd991144c4d47ab3b259e9a9d0b` | `glm-4-32b-0414-coreai` | `decode` | 3 | `benchmarks/raw/20260629-030527-glm-4-32b-0414-coreai` |
| `redhillsmediafl/rhm-gpt-oss-20b-caix` | `ae08b1c0dc03b6ddc53901adaa29e97d23b1cfdb` | `gpt-oss-20b-coreai` | `decode` | 3 | `benchmarks/raw/20260629-002744-gpt-oss-20b-coreai` |
| `redhillsmediafl/rhm-mistral-7b-instruct-v0.3-caix` | `2014f4967181dcab32c75ffb89dbd714f8f89910` | `mistral-7b-instruct-v0.3-coreai` | `decode` | 3 | `benchmarks/raw/20260628-233234-mistral-7b-instruct-v0.3-coreai` |
| `redhillsmediafl/rhm-mistral-nemo-instruct-2407-caix` | `9821a2de1f1029c0352b0d6311c6d399e9aac1fe` | `mistral-nemo-instruct-2407-coreai` | `decode` | 3 | `benchmarks/raw/20260628-234824-mistral-nemo-instruct-2407-coreai` |
| `redhillsmediafl/rhm-mistral-small-instruct-2409-caix` | `0549d42b45a65454bc2f99843deaebeab1587bb1` | `mistral-small-instruct-2409-coreai` | `decode` | 3 | `benchmarks/raw/20260629-000628-mistral-small-instruct-2409-coreai` |
| `redhillsmediafl/rhm-qwen3.6-27b-caix` | `436642eef9fb9fb49f53cafc2d32c0f25a0b175a` | `qwen3.6-27b-coreai` | `decode` | 3 | `benchmarks/raw/20260629-004612-qwen3.6-27b-coreai` |
| `redhillsmediafl/rhm-gemma-4-26b-a4b-caix` | `fede95233c003d99e6db4da433add133ba9458d6` | `gemma-4-26b-a4b-coreai` | `decode` | 3 | `benchmarks/raw/20260629-014239-gemma-4-26b-a4b-coreai` |
| `redhillsmediafl/rhm-gemma-4-31b-it-caix` | `de8554bf1b4c5b26d3bf9eb40cac2b22303eb0e4` | `gemma-4-31b-it-coreai` | `decode` | 3 | `benchmarks/raw/20260629-020515-gemma-4-31b-it-coreai` |

## Manual Or Component Requests

| repo | revision | local dir | request | notes |
|---|---|---|---|---|
| `redhillsmediafl/rhm-gemma-4-26b-a4b-draft-caix` | `d190359eaaaab2e8ce63d86ed41ed9433e555899` | `gemma-4-26b-a4b-draft-coreai` | component; do not test alone | draft component; benchmark with matching target |
| `redhillsmediafl/rhm-gemma-4-31b-it-draft-caix` | `776b4befe3dc5c945b160dec2b6cdcfafb62840c` | `gemma-4-31b-it-draft-coreai` | component; do not test alone | draft component; benchmark with matching target |

## Run Template

Set one row's values:

```bash
REPO=<repo-from-table>
REVISION=<revision-from-table>
NAME=<local-dir-from-table>
```

Install one payload:

```bash
scripts/check-disk-pressure.sh --path /Volumes/SSD --floor-gib 500
mkdir -p models/exports
hf download "$REPO" \
  --revision "$REVISION" \
  --local-dir "models/exports/$NAME"
```

Verify:

```bash
CAIX_BIN=${CAIX_BIN:-.build/release/caix}
MODEL="models/exports/$NAME"

"$CAIX_BIN" inspect --model "$MODEL"
"$CAIX_BIN" run \
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

Report the fields in `docs/TESTING.md`. Send the raw benchmark directory. Remove only the payload
you installed:

```bash
rm -rf "models/exports/$NAME"
scripts/check-disk-pressure.sh --path /Volumes/SSD --floor-gib 500
```
