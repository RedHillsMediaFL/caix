# Tester Requests

Generated from `benchmarks/MANIFEST.tsv`.
Revision source: `benchmarks/revisions.tsv`.
Raw evidence source: `benchmarks/raw`.

No speed claims without raw logs. Use the exact revision in the table. Keep prompts, token budget,
temperature, seed, streaming mode, warmup count, measured run count, and chat-template mode unchanged.

## Ready Benchmark Requests

| repo | revision | local dir | request | notes |
|---|---|---|---|---|
| `redhillsmediafl/rhm-glm-z1-9b-0414-caix` | `c9121d48d4b5161866e1598f40ec1e5822389b25` | `glm-z1-9b-0414-coreai` | load, generation, benchmark | local stdout instability; publish only if stability gate passes |

## Existing Raw Evidence

| repo | revision | local dir | mode | measured runs | raw dir |
|---|---|---|---|---|---|
| `redhillsmediafl/rhm-qwen2.5-0.5b-instruct-caix` | `457592be4e87468a6c64f5567dc3bd46554daa13` | `qwen2.5-0.5b-instruct-coreai` | `decode` | 3 | `benchmarks/raw/20260628-214941-qwen2.5-0.5b-instruct-coreai` |
| `redhillsmediafl/rhm-qwen2.5-3b-instruct-caix` | `e0c019d5a534ec8aea936dfafedb7e00b17c3961` | `qwen2.5-3b-instruct-coreai` | `decode` | 3 | `benchmarks/raw/20260628-220044-qwen2.5-3b-instruct-coreai` |
| `redhillsmediafl/rhm-qwen3-0.6b-caix` | `3aa798b1a942fd15be6e5a96bd18b01e52dc6bc4` | `qwen3-0.6b-coreai` | `decode` | 3 | `benchmarks/raw/20260628-214406-qwen3-0.6b-coreai` |
| `redhillsmediafl/rhm-qwen3-1.7b-caix` | `bf04650aef9a5e325f1ba05c52866b0363f56c29` | `qwen3-1.7b-coreai` | `decode` | 3 | `benchmarks/raw/20260629-072107-qwen3-1.7b-coreai` |
| `redhillsmediafl/rhm-qwen3-4b-caix` | `bc5b7f1b866dee97270d4e1d7bdbdd30e48aa641` | `qwen3-4b-coreai` | `decode` | 3 | `benchmarks/raw/20260628-221149-qwen3-4b-coreai` |
| `redhillsmediafl/rhm-qwen3-8b-caix` | `e46668067aff8efb89e87469269c120073511136` | `qwen3-8b-coreai` | `decode` | 3 | `benchmarks/raw/20260628-222458-qwen3-8b-coreai` |
| `redhillsmediafl/rhm-qwen3-14b-caix` | `62a1e52fd5a5b9c1b241243cb0903d80fb5ad624` | `qwen3-14b-coreai` | `decode` | 3 | `benchmarks/raw/20260628-223516-qwen3-14b-coreai` |
| `redhillsmediafl/rhm-ornith-1.0-9b-caix` | `06656dec7e8165ff714729307ed75eabc3b8c1b5` | `ornith-1.0-9b-coreai` | `decode` | 3 | `benchmarks/raw/20260629-033807-ornith-1.0-9b-coreai` |
| `redhillsmediafl/rhm-qwythos-9b-caix` | `d4783ebca55aa4bf52d9bb4b254266ac5260d9c2` | `qwythos-9b-coreai` | `decode` | 3 | `benchmarks/raw/20260628-224859-qwythos-9b-coreai` |
| `redhillsmediafl/rhm-glm-4-9b-0414-caix` | `c759b1583693b3f051e62fd7082e4bc538ebb72c` | `glm-4-9b-0414-coreai` | `decode` | 3 | `benchmarks/raw/20260628-230705-glm-4-9b-0414-coreai` |
| `redhillsmediafl/rhm-glm-4-32b-0414-caix` | `59bddf6e8d498fd991144c4d47ab3b259e9a9d0b` | `glm-4-32b-0414-coreai` | `decode` | 3 | `benchmarks/raw/20260629-030527-glm-4-32b-0414-coreai` |
| `redhillsmediafl/rhm-gpt-oss-20b-caix` | `ae08b1c0dc03b6ddc53901adaa29e97d23b1cfdb` | `gpt-oss-20b-coreai` | `decode` | 3 | `benchmarks/raw/20260629-002744-gpt-oss-20b-coreai` |
| `redhillsmediafl/rhm-mistral-7b-instruct-v0.3-caix` | `2014f4967181dcab32c75ffb89dbd714f8f89910` | `mistral-7b-instruct-v0.3-coreai` | `decode` | 3 | `benchmarks/raw/20260628-233234-mistral-7b-instruct-v0.3-coreai` |
| `redhillsmediafl/rhm-mistral-nemo-instruct-2407-caix` | `9821a2de1f1029c0352b0d6311c6d399e9aac1fe` | `mistral-nemo-instruct-2407-coreai` | `decode` | 3 | `benchmarks/raw/20260628-234824-mistral-nemo-instruct-2407-coreai` |
| `redhillsmediafl/rhm-mistral-small-instruct-2409-caix` | `0549d42b45a65454bc2f99843deaebeab1587bb1` | `mistral-small-instruct-2409-coreai` | `decode` | 3 | `benchmarks/raw/20260629-000628-mistral-small-instruct-2409-coreai` |
| `redhillsmediafl/rhm-mixtral-8x7b-instruct-caix` | `ea180189c4266d8a0dde4e3238cf959789c0504f` | `mixtral-8x7b-instruct-coreai` | `decode` | 3 | `benchmarks/raw/20260629-042411-mixtral-8x7b-instruct-coreai` |
| `redhillsmediafl/rhm-qwen3.6-27b-caix` | `436642eef9fb9fb49f53cafc2d32c0f25a0b175a` | `qwen3.6-27b-coreai` | `decode` | 3 | `benchmarks/raw/20260629-004612-qwen3.6-27b-coreai` |
| `redhillsmediafl/rhm-gemma-4-26b-a4b-caix` | `fede95233c003d99e6db4da433add133ba9458d6` | `gemma-4-26b-a4b-coreai` | `decode` | 3 | `benchmarks/raw/20260629-014239-gemma-4-26b-a4b-coreai` |
| `redhillsmediafl/rhm-gemma-4-31b-it-caix` | `de8554bf1b4c5b26d3bf9eb40cac2b22303eb0e4` | `gemma-4-31b-it-coreai` | `decode` | 3 | `benchmarks/raw/20260629-020515-gemma-4-31b-it-coreai` |
| `redhillsmediafl/rhm-gemma-4-26b-a4b-mtp-caix` | `98793a039e2a548f04668acb4c254f57ff16f145` | `gemma-4-26b-a4b-mtp-coreai` | `eagle-mtp` | 3 | `benchmarks/raw/20260629-050957-gemma-4-26b-a4b-mtp-coreai-eagle-mtp` |
| `redhillsmediafl/rhm-qwen3-4b-mtp-caix` | `4190150f7a47b113d36ea679c4541d95a21ce3f6` | `qwen3-4b-mtp-coreai` | `speculative` | 3 | `benchmarks/raw/20260629-045620-qwen3-4b-mtp-coreai` |

## Manual Or Component Requests

| repo | revision | local dir | request | notes |
|---|---|---|---|---|
| `redhillsmediafl/rhm-gemma-4-12b-it-unified-caix` | `721d1b798510f6cfcb1608e94ab660ba7d268005` | `gemma4-12b-it-unified-staged-4bit-ctx128-6x8` | distributed hardware smoke | staged manifest; run distributed hardware smoke on 64 GB Studio plus 32 GB MacBook |
| `redhillsmediafl/rhm-gemma-4-12b-staged-caix` | `3ae61884e5bf8ef459c2f0eb4a1c4a1cc2562d3b` | `gemma4-12b-staged-4bit-ctx128-2x24` | distributed hardware smoke | staged manifest; run distributed hardware smoke on 64 GB Studio plus 32 GB MacBook |
| `redhillsmediafl/rhm-gemma-4-e2b-it-staged-caix` | `7b4cb1da35402f25dc1269ef9eebf3f184257dba` | `gemma4-e2b-it-staged-4bit-ctx128-2x` | distributed hardware smoke | staged manifest; run distributed hardware smoke on 64 GB Studio plus 32 GB MacBook |
| `redhillsmediafl/rhm-gemma-4-e2b-staged-caix` | `d5bf65fe72faeb343bf9247ffb8dc9a4bf279cb1` | `gemma4-e2b-staged-4bit-ctx128-2x` | distributed hardware smoke | staged manifest; run distributed hardware smoke on 64 GB Studio plus 32 GB MacBook |
| `redhillsmediafl/rhm-gemma-4-e4b-it-staged-caix` | `bba60d99bdf1e119367b9166938c8c3ea82d1c54` | `gemma4-e4b-it-staged-4bit-ctx128-2x21` | distributed hardware smoke | staged manifest; run distributed hardware smoke on 64 GB Studio plus 32 GB MacBook |
| `redhillsmediafl/rhm-gemma-4-e4b-staged-caix` | `a0e35632c3600aeb3be62507948cb2d66c67d4e1` | `gemma4-e4b-staged-4bit-ctx128-2x21` | distributed hardware smoke | staged manifest; run distributed hardware smoke on 64 GB Studio plus 32 GB MacBook |
| `redhillsmediafl/rhm-gemma-4-26b-a4b-it-staged-caix` | `abf7ad43a38d9132e5b2b2f90d39c2bc261658c6` | `gemma4-26b-a4b-it-staged-4bit-ctx128-5x6` | distributed hardware smoke | staged manifest; run distributed hardware smoke on 64 GB Studio plus 32 GB MacBook |
| `redhillsmediafl/rhm-gemma-4-31b-it-staged-caix` | `c171bcc3df5d68afdc74a019046b784141a3f3b1` | `gemma4-31b-it-staged-4bit-ctx128-6x10` | distributed hardware smoke | staged manifest; run distributed hardware smoke on 64 GB Studio plus 32 GB MacBook |
| `redhillsmediafl/rhm-gemma-4-31b-it-mtp-caix` | `7bfec529dd5a9749588840a8b865c8b9937f454e` | `gemma-4-31b-it-mtp-coreai` | blocked; do not test | draft graph is standard two-input assistant; rebuild package with dependent EAGLE draft |
| `redhillsmediafl/rhm-gemma-4-26b-a4b-draft-caix` | `b1d92267c10b5542c7b3bfdebd18f3814a19c37b` | `gemma-4-26b-a4b-draft-coreai` | component; do not test alone | draft component; benchmark with matching target |
| `redhillsmediafl/rhm-gemma-4-31b-it-draft-caix` | `30c425f86bc123e9b95ddd783fba3398ab4a5604` | `gemma-4-31b-it-draft-coreai` | component; do not test alone | draft component; benchmark with matching target |

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

Report the fields in `docs/TESTING.md`. Send the raw benchmark directory. Remove only the payload
you installed:

```bash
scripts/remove-export.sh "$NAME"
scripts/check-disk-pressure.sh --path /Volumes/SSD --floor-gib 500
```
