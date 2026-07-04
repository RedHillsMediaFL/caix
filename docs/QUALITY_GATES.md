# Quality Gates

Quality evidence is separate from speed evidence. A model can have publishable tok/s and still be
blocked from a model-card quality claim, catalog promotion, or "ready to test" label if the relevant
quality gate below is missing.

The machine-readable gate list is `benchmarks/QUALITY_GATES.tsv`. Keep this page and that TSV in sync
with `scripts/check-quality-gates.sh`.

## Quantized Language Models

Scope: default 4-bit bundles, sub-4-bit experiments, and mixed-precision ladder variants.

Required evidence:

- `quant_ppl_default_4bit`: default 4-bit candidates compare against the same-source bf16/f16
  reference with sequential Core AI logits. Relative perplexity delta must be at most 2%.
- `quant_ppl_ladder_variant`: sub-4-bit or mixed-precision ladder variants compare against the same
  reference. Relative perplexity delta must be at most 5%, and the variant must also show at least a
  15% measured memory or speed gain before promotion.
- `quant_task_eval`: every promoted quantized bundle must pass a fixed task set with the
  `math_reasoning` slice reported separately. Knowledge-only checks are not enough because
  sub-4-bit quality loss can show up in reasoning before it shows up in broad perplexity or knowledge
  prompts.

The fixed task fixture is `quality/quant_tasks_v0.tsv`: 100 prompts with `knowledge`,
`math_reasoning`, and `instruction_following` slices. `scripts/check-quant-eval-contract.sh` verifies
the fixture shape and validates a synthetic raw run against
`scripts/quant-eval/validate-run.py`. This is a contract check only; it is not model-quality
evidence.

Output contract for a real run:

| artifact | required contents |
|---|---|
| `quality/raw/<run>/metadata.json` | caix commit, model repo, model revision, reference bundle, candidate bundle, hardware, OS build, exact command |
| `quality/raw/<run>/quant_ppl.json` | token count, reference perplexity, candidate perplexity, relative delta, pass/fail |
| `quality/raw/<run>/quant_tasks.tsv` | prompt id, slice, expected/graded answer, candidate answer, pass/fail |
| `quality/raw/<run>/summary.json` | gate ids, aggregate scores, thresholds, final decision |

The validator cross-checks `summary.json` against the raw rows. A failed task row can be retained as
blocked evidence, but `final_decision: pass` requires every fixed-task row and the perplexity gate to
pass, with per-slice aggregate counts matching `quant_tasks.tsv`.

Do not use the pipelined text path for perplexity until it can emit full logits. Use a sequential
Core AI logit path or another audited offline evaluator with equivalent logits. Speed is intentionally
irrelevant for the perplexity pass/fail result.

Release labels:

- `pass`: quality evidence exists for the exact model revision and candidate bundle.
- `needs-test`: runnable bundle exists, but one or more required quality artifacts are missing.
- `blocked`: the bundle fails a gate or the reference path is unavailable.
- `comparator-only`: applies to 2-bit/QAT-required ideas that are not reachable through the current
  convert-only Core AI export path.

## Diffusion Models

Scope: diffusion language bundles and any serving route that emits block-diffusion completions.

Required evidence:

- `diffusion_block_quality`: fixed prompt outputs must be captured against an HF reference or a
  verified bf16 Core AI reference. The rubric must include text completion, instruction following, and
  `math_reasoning` prompts.
- `diffusion_api_contract`: before serving is documented, choose and test the API behavior:
  v1 is non-streaming-only. Do not imply token-by-token streaming parity for block diffusion. Committed
  block SSE stays deferred until that event path is implemented and captured.

The API decision is pinned in `quality/diffusion_api_contract_v0.json` and explained in
[DIFFUSION_API.md](DIFFUSION_API.md). `scripts/check-diffusion-api-contract.sh` validates the contract
without loading a model.

The fixed diffusion prompt fixture is `quality/diffusion_prompts_v0.tsv`: 30 prompts split evenly
across `text_completion`, `math_reasoning`, and `instruction_following`. `scripts/check-diffusion-quality-contract.sh`
verifies the fixture shape and validates a synthetic raw run against
`scripts/diffusion-eval/validate-run.py`. This is a contract check only; it is not diffusion quality
evidence.

Output contract for a real run:

| artifact | required contents |
|---|---|
| `quality/raw/<run>/metadata.json` | caix commit, model repo, model revision, reference path, candidate bundle, hardware, OS build, exact command |
| `quality/raw/<run>/diffusion_quality.tsv` | prompt id, slice, reference output, candidate output, rubric result, rater notes |
| `quality/raw/<run>/diffusion_api.json` | selected API mode, request/response examples, streaming decision, pass/fail |
| `quality/raw/<run>/summary.json` | gate ids, aggregate results, final decision |

The validator cross-checks `summary.json` against the raw prompt rows. A failed or `needs_review`
rubric row can be retained as blocked evidence, but `final_decision: pass` requires every fixed prompt
and the API contract artifact to pass, with per-slice aggregate counts matching `diffusion_quality.tsv`.

Diffusion quality is not a speed benchmark and not a staged-parity substitute. It gates serving and
publication language for diffusion bundles only.
