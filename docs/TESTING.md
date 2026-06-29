# External Testing

Use this when testing a converted `redhillsmediafl/*-caix` bundle on hardware we do not have.

Do not submit speed claims without raw logs. Do not compare runs that use different prompts, token
budgets, temperature, streaming mode, or chat-template mode.

## Pick a Target

Prefer repos that are not already verified on the host you are using, or repos missing measured raw
benchmark logs. Check the model card and `benchmarks/MANIFEST.tsv` first.

Generate a current request sheet when assigning external tests:

```bash
scripts/check-benchmark-coverage.sh
scripts/check-hf-collections.sh
scripts/check-conversion-ledger.sh
scripts/generate-tester-requests.sh \
  --revisions benchmarks/revisions.tsv \
  --out docs/TESTER_REQUESTS.md
scripts/check-tester-requests.sh \
  --revisions benchmarks/revisions.tsv
```

Draft repos are components. Test them only with the matching target repo and the command documented
on the model card.

MTP repos are target-plus-draft packages. Report target-only and MTP/speculative results separately.

## Record the Exact Revision

Use the exact model repo commit in every report:

```bash
REPO=redhillsmediafl/rhm-qwen3-4b-caix
hf models info "$REPO" --format json > model-info.json
```

Record the commit SHA from `model-info.json` or from the model page.

For a manifest-wide run, use the metadata collector:

```bash
scripts/collect-model-revisions.sh \
  --out benchmarks/revisions.tsv \
  --details benchmarks/revisions-details.tsv
```

## Install the Bundle

Use a clean local directory:

```bash
REPO=redhillsmediafl/rhm-qwen3-4b-caix
REVISION=<model-repo-commit>
NAME=qwen3-4b-coreai

mkdir -p models/exports
hf download "$REPO" \
  --revision "$REVISION" \
  --local-dir "models/exports/$NAME"
```

Do not install multiple model payloads at once unless you have checked free disk first.

## Verify Load and Generation

Use the release binary when available:

```bash
CAIX_BIN=${CAIX_BIN:-.build/release/caix}
MODEL=models/exports/qwen3-4b-coreai

"$CAIX_BIN" inspect --model "$MODEL"
"$CAIX_BIN" run \
  --model "$MODEL" \
  --prompt "Name one primary color." \
  --max-tokens 32 \
  --temperature 0 \
  --verbose
```

Pass condition:

- `inspect` completes without a model-contract error.
- `run` loads the bundle and emits text.
- The stderr summary includes prompt tokens, generated tokens, stop reason, load seconds, prefill
  seconds, decode seconds, and decode tok/s.

Fail condition:

- The process exits nonzero.
- Core AI reports a model-contract, specialization, memory, or shape error.
- Output is empty after a successful load.

## Benchmark

Use the shared runner:

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

For an installed classic speculative package with a `draft/` bundle, use the normal runner with
`--draft`:

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

For an installed EAGLE/MTP package with `eagle_target.aimodel` and `eagle_draft.aimodel`, use the
EAGLE runner:

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

Use the standalone target row for target-only numbers. Keep target+draft package rows separate.

Keep the full output directory from `benchmarks/raw/`. Do not copy only the final number.

For a suite run, write or collect a TSV with exact revisions:

```bash
printf '%s\t%s\n' "$REPO" "$REVISION" > benchmarks/revisions.tsv
scripts/collect-model-revisions.sh --out benchmarks/revisions.tsv
scripts/benchmark-suite.sh \
  --exports models/exports \
  --revisions benchmarks/revisions.tsv \
  --warmup 1 \
  --runs 3
```

## Report

Include these fields:

| field | value |
|---|---|
| model repo | `redhillsmediafl/...-caix` |
| model repo revision | exact commit SHA |
| local bundle path | `models/exports/...` |
| caix commit | `git rev-parse HEAD` |
| caix binary | path used for `inspect`, `run`, and benchmark |
| macOS build | `sw_vers -productVersion` and `sw_vers -buildVersion` |
| hardware | `sysctl -n machdep.cpu.brand_string` |
| memory bytes | `sysctl -n hw.memsize` |
| verification command | exact command |
| verification result | pass or fail, with stderr on fail |
| benchmark raw directory | `benchmarks/raw/<timestamp>-<name>/` |
| benchmark report row | output from `scripts/benchmark-report.sh` when available |

Send raw stdout/stderr files for failed runs. For successful benchmark rows, send the raw directory
or an archive of it.

Before editing public docs or model cards, run:

```bash
scripts/check-public-copy.sh README.md docs web Formula
```

For a model card draft, pass its `README.md` path to the same script.
For live RHM cards, run:

```bash
scripts/check-hf-model-cards.sh
```

## Cleanup

After testing, remove only the payload you installed:

```bash
rm -rf "models/exports/$NAME"
```

Check free disk before starting another test:

```bash
scripts/check-disk-pressure.sh --path /Volumes/SSD --floor-gib 500
du -sh models/exports benchmarks/raw 2>/dev/null
```
