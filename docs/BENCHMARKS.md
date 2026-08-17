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

## Decode Ceiling Model

Use the no-load ceiling calculator before interpreting speed work:

```bash
scripts/perf/ceiling.py --out benchmarks/CEILINGS.md
scripts/check-ceilings.sh
```

The calculator reads `benchmarks/CEILING_ASSUMPTIONS.tsv` plus `benchmarks/MANIFEST.tsv` and writes
`benchmarks/CEILINGS.md`. It does not load models, benchmark, download, or contact external services.
The default usable-bandwidth assumption is 550 GiB/s, matching the current planning midpoint for the
M1 Ultra. Rows without reviewed active-weight bytes remain `missing_estimate`; do not quote
percent-of-ceiling for those rows. The checker rejects stale generated output, assumption rows that
drift from the manifest's repo/local-dir/kind/mode tuple, duplicate assumptions, invalid evidence
labels, and non-positive active-weight values.

To join a completed benchmark report and see utilization against the ceiling:

```bash
scripts/perf/ceiling.py \
  --benchmark-report benchmarks/raw/<timestamp>-suite/report.tsv \
  --format tsv
```

Treat the output as a planning denominator. It becomes publishable only when the underlying active
weight estimate is measured or otherwise reviewed and the benchmark report itself satisfies the raw
evidence gates.

To add no-load local bundle-size context, pass a bundle root:

```bash
scripts/perf/ceiling.py \
  --bundle-root models/exports \
  --format tsv
```

This reports logical sizes for discovered `.aimodel` packages while keeping them separate from
`active_weight_gib`. For MoE and speculative packages, total asset size is not active bytes per
decode token.

Run `scripts/check-benchmark-coverage.sh` before assigning tests or collecting revisions. It compares
the manifest with live `redhillsmediafl/rhm-*-caix` Hub metadata and fails if a converted repo is
missing from benchmark coverage. It also rejects non-canonical manifest modes; use `decode`,
`speculative`, `eagle-mtp`, or `manual`.

Run `scripts/check-hf-collections.sh` after changing the manifest or Hugging Face collections. It
fails if a manifest repo is missing from the public family collections or if a collection note uses
speed/fluff wording. The no-network fixture regression is `scripts/check-hf-collections-contract.sh`;
publication gates run it so stale collection-note wording such as `pending` cannot re-enter local
release inputs without using the Hub.

Run `scripts/check-hf-model-cards.sh` before uploading card edits. It fetches only live
`README.md` files for manifest repos, requires the plain support link, applies the public-copy
guard, and enforces the production card contract: `library_name: caix`, `base_model`, a small RHM
logo, `Install & run`, `At a glance` specs, short status, `## License`, and the open-source footer.
Internal evidence numbers, build details, and test-device notes stay in `benchmarks/MANIFEST.tsv`,
revision tables, and `docs/CONVERSION_LEDGER.tsv`, not public cards. MTP/speculative cards must
identify the target+draft package and matching standalone target context in user-facing language.
The public-copy guard rejects positive structured-output support claims until release-reviewed model
smoke evidence deliberately lifts that copy gate, and the publication gate runs local fixtures that
prove the positive-claim and raw-speed copy checks still fail closed. Structured-output smoke
artifacts are validated by `scripts/check-structured-output-evidence.sh`; the publication gate runs
`scripts/check-structured-output-evidence-contract.sh` so the validator fails closed without requiring
a model load. It also runs `scripts/check-structured-output-release-readiness-contract.sh`, which
self-tests the strict xgrammar pin/vendor blocker without making current development dependencies a
publication-gate failure, and now rejects release evidence that only succeeds by `maxTokens` or
`contextLimit` truncation.
The same public-copy contract also fixture-tests prefix-cache wording. Exact continuation reuse may
apply in loaded CoreAILM fast handles, but general prompt caching is unsupported in public copy, and
partial or semantic prefix reuse is blocked until a release gate proves it.
It also rejects broad media-capability copy. Gemma 4 image+text wording must be tied to a verified
runtime bundle plus serving evidence; audio, video, multipart, and unsupported backends must remain
framed as clean 400/503 errors.
Serving-path label drift is blocked in the same fixture: staged artifacts are distributed artifacts,
not the single-device fast path, and 4-bit monolithic bundles must not claim fp16 1:1.
Markdown-wrapped speed numbers and `x faster` wording are covered by the public-copy fixture as well;
they need publishable raw benchmark evidence or explicit internal/not-publishable framing.
Tester request sheets are checked before export-cleanliness and must distinguish target-only from
target+draft speculative/EAGLE evidence whenever the manifest contains those rows.
Distributed ready-to-test and staged upload-ready copy is blocked until two-machine hardware evidence
and sign-off exist; Studio-only parity evidence is not enough for those claims.
RDMA/TB5 support, speedup, or tensor-parallel copy is also blocked until a Thunderbolt 5 pair
produces captured negotiation evidence and token-accurate raw evidence.
Use `--cards-dir <dir>` with `${repo//\//__}.README.md` fixtures for local contract tests.
The local publication gate runs `scripts/check-hf-model-card-contract.sh`, which exercises this
contract without touching the Hub.
It also runs `scripts/check-publication-gates-contract.sh`, a no-load self-test that keeps
`--distributed` wired to strict tracked distributed evidence.

Run `scripts/check-token-handling.sh` before committing Hub automation or docs. It rejects direct
HF token env reads, Bearer auth headers, and token argv patterns.

Run `scripts/check-quality-gates.sh` after changing quality-gate docs or benchmark metadata. It keeps
`docs/QUALITY_GATES.md` and `benchmarks/QUALITY_GATES.tsv` aligned with the required quant and
diffusion promotion gates. This is a no-load static check; real quality runs write evidence under
`quality/raw/<run>/`.

Run `scripts/check-quant-eval-contract.sh` after changing the fixed quant task fixture or quant raw
evidence contract. It validates `quality/quant_tasks_v0.tsv`, the synthetic raw-run shape, and the
no-load validator under `scripts/quant-eval/`.

Run `scripts/check-diffusion-api-contract.sh` after changing diffusion serving docs or quality-gate
metadata. The v1 contract is non-streaming-only and lives in `quality/diffusion_api_contract_v0.json`;
block SSE remains a separate future implementation and evidence item.

Run `scripts/check-diffusion-quality-contract.sh` after changing the fixed diffusion prompt fixture or
raw evidence contract. It validates `quality/diffusion_prompts_v0.tsv`, the synthetic raw-run shape,
and the no-load validator under `scripts/diffusion-eval/`.

Run `scripts/check-pipelined-kv-guardrail.sh` after changing the CoreAILM fast path. It keeps
`PipelinedLLM` on `.auto`/growing KV behavior and rejects blanket `.fixedSize` or ad hoc `kvCacheSize`
changes until a request-bounded prompt+max+headroom policy is reviewed.

Run `scripts/check-conversion-ledger.sh` after changing `models/registry.json`,
`docs/CONVERSION_LEDGER.tsv`, or the benchmark manifest. It keeps active conversion lanes explicit:
published, component-only, or blocked with a next step. Published staged repos must name the staged
or distributed hardware follow-up, MTP/speculative repos must name the target/draft benchmark or
rebuild action, and draft repos must stay labeled as components that require a matching target. Run
`scripts/check-conversion-ledger-contract.sh` when changing that ledger contract.

Run `scripts/audit-conversion-gaps.sh --out docs/CONVERSION_GAP_AUDIT.tsv` to refresh source
metadata for active conversion lanes. It reads Hub metadata only.
Run `scripts/check-conversion-gap-audit.sh` before committing the refreshed TSV.

Run `scripts/check-tester-requests.sh` after changing the manifest, raw benchmark logs, or
`docs/TESTER_REQUESTS.md`. It regenerates the request sheet and fails if the committed sheet is
stale.
The publication gate also runs `scripts/check-tester-requests-contract.sh`, which fixture-tests that
valid raw evidence suppresses tester requests only when it is admissible: out-of-tree raw fixtures
count, but untracked in-repo raw folders do not. The fixture also pins staged rows to distributed
hardware smoke, blocked rows to do-not-test wording, draft rows to component-only wording, and MTP
rows to target+draft benchmark wording.

Create `benchmarks/revisions.tsv` before a publishable run:

```bash
scripts/collect-model-revisions.sh \
  --out benchmarks/revisions.tsv \
  --details benchmarks/revisions-details.tsv
scripts/check-model-revisions.sh
```

The revisions file is a local run artifact and is ignored by default. It contains:

```text
redhillsmediafl/rhm-qwen3-4b-caix<TAB><model-repo-commit>
```

Non-dry-run suite rows refuse to measure without a 40-character model repo revision.
`scripts/check-model-revisions.sh` also keeps the revision table exact relative to
`benchmarks/MANIFEST.tsv`, with no missing, stale, duplicate, or malformed rows. Re-run the
collection step immediately before measuring if any model repo was updated.

Runtime dependency evidence is tracked separately from model repo revisions:

```bash
scripts/collect-dependency-evidence.sh --out benchmarks/DEPENDENCY_EVIDENCE.tsv
scripts/check-dependency-evidence.sh
```

This records the exact resolved `coreai-models` and `xgrammar` SwiftPM revisions plus the Core AI
Python export dependency versions declared by the checked-out `coreai-models` package. Update it
before public benchmark tables, release notes, or structured-output capability claims. For structured
output, `scripts/check-structured-output-release-readiness.sh --evidence <smoke.json>` additionally
rejects public-readiness mode while `xgrammar` remains branch-based; `--allow-branch-dependencies` is
development reporting only.
The publication gate runs `scripts/check-dependency-evidence-contract.sh`, which fixture-tests
collector/checker behavior for current evidence, stale evidence, missing SwiftPM pins, and ranged
instead of exact Core AI Python dependencies without using live dependency state.

## Report Gate

Create a report from a completed suite run:

```bash
scripts/benchmark-report.sh \
  --suite benchmarks/raw/<timestamp>-suite \
  --out benchmarks/reports/<timestamp>.tsv
```

The report script reads the suite summary and each measured model's raw `summary.tsv` and
`metadata.txt`. It refuses missing raw logs, raw paths outside the suite's raw evidence root, missing
captured stdout/stderr, and failed measured rows. Rows without a recorded model repo revision are
marked `publishable=no`; do not copy those numbers into public docs. The report includes
`benchmark_mode` and refuses suite/model setting drift; do not compare rows unless the mode and prompt
settings match.

Run `scripts/check-benchmark-raw.sh` before committing raw benchmark logs. It checks clean run-start
git status for new or changed raw dirs, pinned model repo revisions, recorded caix commits,
raw output path containment, suite/model metadata consistency, measured row counts, failed rows, and
deterministic measured stdout. It also pins the suite and model `summary.tsv` schemas before reading
positional fields; `scripts/check-benchmark-raw-contract.sh` fixture-tests that stale or reordered
summary schemas fail closed without loading a model.
Publication gates run it with `--require-tracked` so public checks cannot pass from local probe logs
that were not committed.

Run `scripts/check-benchmark-gaps.sh` to list eligible manifest rows that still lack committed
measured raw evidence. Use `--strict` only when the current release must have no eligible benchmark
gaps. The publication gate runs `scripts/check-benchmark-gaps-contract.sh`, which fixture-tests the
non-strict report path, strict failure path, all-clear path, and repo+benchmark-mode matching so
decode evidence cannot satisfy a speculative/MTP row.

Export cleanup is checked after local evidence gates. `scripts/check-export-cleanliness.sh --report`
prints the non-destructive cleanup plan, and `scripts/check-export-cleanliness-contract.sh`
fixture-tests clean, local-payload, tracked-payload, missing-`.gitkeep`, and report-mode behavior
without touching real bundles.

Before publishing docs, model cards, or benchmark reports, run:

```bash
scripts/check-publication-gates.sh --hub
```

Use `--strict-evidence` for final publication review when distributed parity evidence is part of the
claim set. It requires distributed evidence summaries and raw logs to be tracked, mirroring the raw
benchmark `--require-tracked` gate. `--distributed` enables the same strict evidence check
automatically.

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
- RHM model cards must also avoid quality claims until the relevant gates in
  [QUALITY_GATES.md](QUALITY_GATES.md) have exact raw evidence.
- `benchmarks/MANIFEST.tsv` is the current RHM caix benchmark coverage list.
- `docs/CONVERSION_LEDGER.tsv` is the current conversion lane status list.
- Gemma 3 is blocked until Hugging Face access is approved.
- Qwen3-32B and Qwen3-30B-A3B are skipped on this 64 GB host by conversion fit-check.
