# Quant Eval Contract

This directory holds no-load contract tooling for the P3 quant-quality gate. It does not run Core AI,
load bundles, benchmark, or contact external services.

The fixed task fixture is `quality/quant_tasks_v0.tsv`. A promoted quantized language bundle must run
all 100 prompts and report the `math_reasoning` slice separately.

Future heavy evaluator command shape:

```bash
scripts/quant-eval/run.sh \
  --reference-bundle <bf16-or-f16-reference> \
  --candidate-bundle <converted-candidate> \
  --tasks quality/quant_tasks_v0.tsv \
  --out quality/raw/<run>
```

Real runs must write:

- `metadata.json`
- `quant_ppl.json`
- `quant_tasks.tsv`
- `summary.json`

Use `scripts/quant-eval/validate-run.py --run quality/raw/<run>` after a heavy run to validate the
raw evidence shape before making any model-card or catalog claim.

The validator keeps failed evidence usable when `summary.json` says `blocked`, but a
`final_decision` of `pass` must agree with the raw rows: all fixed-task rows pass, per-slice
`passed`/`total` counts match `quant_tasks.tsv`, and `quant_ppl.json` has `pass: true`.
