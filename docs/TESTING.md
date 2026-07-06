# External Testing

Use this when testing a converted `redhillsmediafl/*-caix` bundle on hardware we do not have.

Do not submit speed claims without raw logs. Do not compare runs that use different prompts, token
budgets, temperature, seed, streaming mode, or chat-template mode.

Catalog publication and submission rules are in [CATALOG_SUBMISSIONS.md](CATALOG_SUBMISSIONS.md).
Quality promotion gates are in [QUALITY_GATES.md](QUALITY_GATES.md).

## Pick a Target

Prefer repos that are not already verified on the host you are using, or repos missing measured raw
benchmark logs. Check the model card and `benchmarks/MANIFEST.tsv` first.

Generate a current request sheet when assigning external tests:

```bash
scripts/check-publication-gates.sh --hub
scripts/check-token-handling.sh
scripts/generate-tester-requests.sh \
  --revisions benchmarks/revisions.tsv \
  --out docs/TESTER_REQUESTS.md
scripts/check-tester-requests.sh \
  --revisions benchmarks/revisions.tsv
```

For final publication review with distributed parity evidence, add `--strict-evidence` to
`scripts/check-publication-gates.sh`; `--distributed` enables the same strict evidence check
automatically. It requires distributed evidence summaries and raw logs to be tracked instead of
merely present in the working tree.

Draft repos are components. Test them only with the matching target repo and the command documented
on the model card.

MTP repos are target-plus-draft packages. Report target-only and MTP/speculative results separately.
The generated tester request sheet must carry that warning whenever the manifest has speculative or
EAGLE/MTP rows; `scripts/check-tester-requests.sh` enforces it before publication.

## Record the Exact Revision

Use the exact model repo commit in every report:

```bash
REPO=redhillsmediafl/rhm-qwen3-4b-caix
export HF_HOME=${HF_HOME:-/Volumes/SSD/hf-cache}
hf models info "$REPO" --format json > model-info.json
```

Record the commit SHA from `model-info.json` or from the model page.

For a manifest-wide run, use the metadata collector:

```bash
scripts/collect-model-revisions.sh \
  --out benchmarks/revisions.tsv \
  --details benchmarks/revisions-details.tsv
scripts/check-model-revisions.sh
```

## Install the Bundle

Use a clean local directory:

```bash
REPO=redhillsmediafl/rhm-qwen3-4b-caix
REVISION=<model-repo-commit>
NAME=qwen3-4b-coreai

export HF_HOME=${HF_HOME:-/Volumes/SSD/hf-cache}
scripts/check-disk-pressure.sh --path /Volumes/SSD --floor-gib 500
mkdir -p models/exports
hf download "$REPO" \
  --revision "$REVISION" \
  --local-dir "models/exports/$NAME"
```

Do not install multiple model payloads at once unless you have checked free disk first.

## Verify Load and Generation

Use the release binary when available:

```bash
caix_bin=${caix_bin:-.build/release/caix}
MODEL=models/exports/qwen3-4b-coreai

"$caix_bin" inspect --model "$MODEL"
"$caix_bin" run \
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

### Hybrid qwen3_5 Full-Context Gate

`qwen3_5` and `qwen3_5_moe` bundles use hybrid recurrent/full-attention state, not a standard
all-layer KV cache. A full-context claim for these models needs stronger evidence than allocation
or a high-position smoke:

- Short-sequence parity must match a deterministic HF/token oracle.
- The bundle must allocate its native KV capacity and serve a long prompt without memory pressure.
- Publication as "full native context" additionally requires a practical contiguous replay gate
  from token 0 through the claimed context, or an explicitly narrower card/status label approved in
  the conversion ledger.

Do not use a synthetic jump to a high `position_range` as evidence. The recurrent/full KV state is
created by replaying the sequence; jumping would leave the cache empty and only exercise position
encoding. Current Qwythos evidence is short HF parity, native 1,048,576 KV-capacity allocation, and
contiguous `TextStagedRealAssetTests/testQwythosLongContextSmoke` gates at
`COREAI_STAGED_PREFILL_CHUNK=128` through 512 tokens (`prefill=24.193s`), 4,096 tokens
(`prefill=183.167s`), and 16,384 tokens (`prefill=735.048s`). The 16,384-token run projects to about
13.1 h for a full 1,048,576-token replay, so Qwythos public sequential context is capped at 16,384
tokens until replay performance improves. That evidence is useful for runtime bring-up, but it is
not enough to publish a full-1M sequential-context claim.

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
scripts/check-model-revisions.sh
scripts/check-dependency-evidence.sh
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
scripts/check-token-handling.sh
scripts/check-public-copy.sh README.md docs web Formula
```

For a model card draft, pass its `README.md` path to the same script.
The public-copy guard allows only the narrow prefix-cache wording that exact continuation reuse may
apply in loaded CoreAILM fast handles. General prompt caching is unsupported in public copy, and
partial or semantic prefix reuse is blocked. Cross-session caches and SSD-persistent KV snapshots
are blocked until a release gate proves them.
The same guard rejects broad media-capability copy. Gemma 4 image+text copy must be tied to a
verified runtime bundle plus serving evidence; audio, video, multipart, and unsupported backends must
still be described as clean 400/503 errors.
Serving-path labels are also guarded: `<model>-staged` is the distributed artifact, not the
single-device fast path, and 4-bit monolithic bundles must not claim fp16 1:1.
Markdown-wrapped speed numbers and `x faster` wording are guarded too. They must either cite
publishable raw benchmark evidence or be explicitly framed as internal/not-publishable evidence.
Distributed ready-to-test and staged upload-ready copy is blocked until two-machine hardware
evidence and sign-off exist; Studio-only parity evidence is not enough for those claims.
RDMA/TB5 support, speedup, or tensor-parallel copy is blocked until a Thunderbolt 5 pair produces
captured negotiation evidence and token-accurate raw evidence.
For live RHM cards, run:

```bash
scripts/check-hf-model-cards.sh
```

For live collection notes, use `scripts/check-hf-collections.sh`. The local fixture regression
`scripts/check-hf-collections-contract.sh` proves manifest coverage and stale-note cleanup without
touching the Hub.

That check reads only card `README.md` files from the Hub. It does not download model payloads.
Cards must carry the production user-facing contract: `library_name: caix`, `base_model`,
small RHM logo, `Install & run`, `At a glance` specs, short status, `## License`, and the
open-source footer. Internal validation numbers, build details, and test-device notes stay in the
manifest, revision tables, and ledger, not public cards. Use
`scripts/check-hf-model-cards.sh --cards-dir <dir>` to run the same contract against local
`${repo//\//__}.README.md` fixtures without touching the Hub.
The local fixture regression is `scripts/check-hf-model-card-contract.sh`.

Structured-output capability claims need real CoreAILM constrained-decoding smoke evidence, not
only request parsing or typecheck evidence:

```bash
CAIX_STRUCTURED_OUTPUT_MODEL=/path/to/coreai-language-bundle \
  scripts/check-structured-output-smoke.sh --output .tmp/structured-output-smoke.json
```

The smoke loads the supplied bundle through the persistent model handle, sends a tiny `json_schema`
request through CoreAILM constrained decoding, parses the generated text as JSON, and verifies it
matches the schema. The current implementation uses CoreAILM's sequential engine variant for
constrained decoding because the pipelined engine samples on GPU and does not expose logits. Keep
public docs and card copy silent until this evidence exists for the release target and publication
gates pass.

Validate captured smoke evidence without loading a model:

```bash
scripts/check-structured-output-evidence.sh .tmp/structured-output-smoke.json
scripts/check-structured-output-evidence-contract.sh
```

The no-load validator parses the captured schema and generated text, verifies that the text is JSON,
checks the generated object against the schema subset used by the smoke, and is exercised by a local
contract fixture inside `scripts/check-publication-gates.sh`. Passing this validator records evidence
shape only; public capability copy still needs release review and explicit sign-off.

Before lifting public `response_format` copy, run the stricter readiness check:

```bash
scripts/check-structured-output-release-readiness.sh \
  --evidence .tmp/structured-output-smoke.json
```

Strict mode also checks dependency provenance and fails while `xgrammar` is still tracked as a branch
dependency. `--allow-branch-dependencies` is only a development/report mode; it is not release
readiness. Release evidence must stop by `eos` or `stopSequence` and record a positive decode rate;
a schema-valid object truncated by `maxTokens` or `contextLimit` is diagnostic evidence, not public
readiness. `scripts/check-structured-output-release-readiness-contract.sh` fixture-tests strict and
development modes inside publication gates without using the live dependency state.

Before a quantized language bundle can move from `needs-test` to a quality claim, run the fixed task
fixture from `quality/quant_tasks_v0.tsv` in the heavy evaluator window and validate the resulting
`quality/raw/<run>/` directory with:

```bash
scripts/quant-eval/validate-run.py --run quality/raw/<run>
```

Diffusion bundles have a separate API contract in `quality/diffusion_api_contract_v0.json`. Until
committed-block SSE is implemented and tested, diffusion serving is non-streaming-only and streaming
requests must not be described as supported.
For diffusion quality evidence, run the fixed prompt fixture from `quality/diffusion_prompts_v0.tsv`
in the heavy evaluator window and validate the resulting `quality/raw/<run>/` directory with:

```bash
scripts/diffusion-eval/validate-run.py --run quality/raw/<run>
```

## Cleanup

After testing, preview cleanup, then remove only the payload you installed:

```bash
scripts/remove-export.sh --dry-run "$NAME"
scripts/remove-export.sh "$NAME"
```

Before publication review, `models/exports` must contain only the tracked `.gitkeep`. Use the
non-destructive report first:

```bash
scripts/check-export-cleanliness.sh --report
```

Publication gates run `scripts/check-export-cleanliness-contract.sh` with temporary fixtures and a
temporary Git index so clean, local-payload, tracked-payload, missing-`.gitkeep`, and report-mode
behavior stay pinned without touching real model payloads.

Check free disk before starting another test:

```bash
scripts/check-disk-pressure.sh --path /Volumes/SSD --floor-gib 500
du -sh models/exports benchmarks/raw 2>/dev/null
```
