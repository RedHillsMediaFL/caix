# Benchmarks

Benchmark numbers are publishable only when the raw run data is kept with the exact model, commit,
machine, and command.

## Rules

- Run no benchmarks while `.agent-heavy-task.lock` exists or while a conversion/upload/verification is active.
- Run publishable benchmark scripts only from a clean git worktree.
- Use the same prompt, token budget, temperature, seed, streaming mode, warmup count, and measured
  run count for every comparable model.
- Record skipped models with the reason: missing local bundle, gated upstream access, host memory fit, runtime failure, or license limit.
- Use median decode tok/s from measured warm runs for public tables. Keep prefill and load time separately.
- Do not mix chat-template runs with `--raw` runs in the same comparison.
- Do not publish micro-benchmark results from `caix bench` as decode tok/s. That command measures forward-pass shape cost only.
- Do not add a model-card benchmark row unless the raw logs for that exact model repo commit exist.

## Default Decode Run

Use this for plain text-generation comparisons unless a model requires a different prompt format:

```bash
scripts/benchmark-model.sh \
  --model models/exports/qwen3-4b-coreai \
  --name qwen3-4b-coreai \
  --repo redhillsmediafl/rhm-qwen3-4b-caix \
  --repo-revision <model-repo-commit> \
  --prompt "Write one factual sentence about local inference on Apple silicon." \
  --max-tokens 128 \
  --warmup 1 \
  --runs 3
```

Output goes under `benchmarks/raw/<timestamp>-<name>/`. That path is ignored by default so bulk
local logs do not get committed by accident. Keep or publish raw run folders deliberately when adding
public numbers. The direct model and EAGLE runners require both `--repo` and a 40-character
`--repo-revision`; use `caix run` for local ad hoc checks that are not benchmark evidence.

## Suite Run

Use the suite manifest to account for every known RHM caix repo:

```bash
scripts/check-benchmark-coverage.sh
scripts/check-hf-collections.sh
scripts/check-token-handling.sh
scripts/benchmark-suite.sh --dry-run
scripts/collect-model-revisions.sh \
  --out benchmarks/revisions.tsv \
  --details benchmarks/revisions-details.tsv
scripts/benchmark-suite.sh \
  --exports models/exports \
  --revisions benchmarks/revisions.tsv \
  --warmup 1 \
  --runs 3
```

The suite reads `benchmarks/MANIFEST.tsv`, does not download models, and writes one row per repo to
`benchmarks/raw/<timestamp>-suite/summary.tsv` with the row's `benchmark_mode`. Installed standalone bundles use
`scripts/benchmark-model.sh`. Classic speculative packages use `scripts/benchmark-model.sh` with
`<bundle>/draft`. EAGLE/MTP packages use `benchmark_mode=eagle-mtp` and `scripts/benchmark-eagle.sh` against
`eagle_target.aimodel`, `eagle_draft.aimodel`, and `tokenizer/`. Missing bundles, missing draft
bundles, and draft-only repos are recorded as skipped with the reason.

Standalone target rows and target+draft package rows are separate rows. Report target-only and
speculative/MTP numbers separately; do not average them.

`--seed` applies to standalone and classic speculative `caix run` rows. Do not use it for EAGLE/MTP
suite rows.

Run `scripts/check-benchmark-coverage.sh` before assigning tests or collecting revisions. It compares
the manifest with live `redhillsmediafl/rhm-*-caix` Hub metadata and fails if a converted repo is
missing from benchmark coverage. It also rejects non-canonical manifest modes; use `decode`,
`speculative`, `eagle-mtp`, or `manual`.

Run `scripts/check-hf-collections.sh` after changing the manifest or Hugging Face collections. It
fails if a manifest repo is missing from the public family collections or if a collection note uses
speed/fluff wording.

Run `scripts/check-hf-model-cards.sh` before uploading card edits. It fetches only live
`README.md` files for manifest repos, requires the plain support link, and applies the
public-copy guard.

Run `scripts/check-token-handling.sh` before committing Hub automation or docs. It rejects direct
HF token env reads, Bearer auth headers, and token argv patterns.

Run `scripts/check-conversion-ledger.sh` after changing `models/registry.json`,
`docs/CONVERSION_LEDGER.tsv`, or the benchmark manifest. It keeps active conversion lanes explicit:
published, component-only, or blocked with a next step.

Run `scripts/audit-conversion-gaps.sh --out docs/CONVERSION_GAP_AUDIT.tsv` to refresh source
metadata for active conversion lanes. It reads Hub metadata only.
Run `scripts/check-conversion-gap-audit.sh` before committing the refreshed TSV.

Run `scripts/check-tester-requests.sh` after changing the manifest, raw benchmark logs, or
`docs/TESTER_REQUESTS.md`. It regenerates the request sheet and fails if the committed sheet is
stale.

Create `benchmarks/revisions.tsv` before a publishable run:

```bash
scripts/collect-model-revisions.sh \
  --out benchmarks/revisions.tsv \
  --details benchmarks/revisions-details.tsv
```

The revisions file is a local run artifact and is ignored by default. It contains:

```text
redhillsmediafl/rhm-qwen3-4b-caix<TAB><model-repo-commit>
```

Non-dry-run suite rows refuse to measure without a 40-character model repo revision. Re-run the
collection step immediately before measuring if any model repo was updated.

## Report Gate

Create a report from a completed suite run:

```bash
scripts/benchmark-report.sh \
  --suite benchmarks/raw/<timestamp>-suite \
  --out benchmarks/reports/<timestamp>.tsv
```

The report script reads the suite summary and each measured model's raw `summary.tsv` and
`metadata.txt`. It refuses missing raw logs and failed measured rows. Rows without a recorded model
repo revision are marked `publishable=no`; do not copy those numbers into public docs. The report
includes `benchmark_mode` and refuses suite/model setting drift; do not compare rows unless the mode
and prompt settings match.

Run `scripts/check-benchmark-raw.sh` before committing raw benchmark logs. It checks clean run-start
git status for new or changed raw dirs, pinned model repo revisions, suite/model metadata consistency,
measured row counts, failed rows, and deterministic measured stdout.
Publication gates run it with `--require-tracked` so public checks cannot pass from local probe logs
that were not committed.

Run `scripts/check-benchmark-gaps.sh` to list eligible manifest rows that still lack committed
measured raw evidence. Use `--strict` only when the current release must have no eligible benchmark
gaps.

Before publishing docs, model cards, or benchmark reports, run:

```bash
scripts/check-publication-gates.sh --hub
```

## Public Table Fields

Use these fields for any published benchmark table:

| field | source |
|---|---|
| model repo | Hugging Face repo id and commit SHA |
| caix commit | `git rev-parse HEAD` |
| hardware | `sysctl -n machdep.cpu.brand_string`, unified memory, macOS build |
| command | exact `caix run` or server request |
| benchmark mode | `decode`, `speculative`, or `eagle-mtp` |
| prompt | exact prompt text or fixture path |
| max tokens | command value |
| temperature | command value |
| seed | command value; blank means no seed |
| mode | chat template or raw; streaming or non-streaming |
| load seconds | caix stderr summary |
| prefill seconds | caix stderr summary |
| decode seconds | caix stderr summary |
| output tokens | caix stderr summary |
| decode tok/s | output tokens divided by decode seconds; use median across measured runs |

## Current Gaps

- RHM model cards intentionally omit benchmark rows until measured public numbers exist.
- `benchmarks/MANIFEST.tsv` is the current RHM caix benchmark coverage list.
- `docs/CONVERSION_LEDGER.tsv` is the current conversion lane status list.
- Gemma 3 is blocked until Hugging Face access is approved.
- Qwen3-32B and Qwen3-30B-A3B are skipped on this 64 GB host by conversion fit-check.
