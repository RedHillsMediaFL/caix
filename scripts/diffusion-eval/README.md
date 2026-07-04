# Diffusion Eval Contract

This directory holds no-load contract tooling for the P2.4 diffusion quality gate. It does not run
Core AI, load bundles, benchmark, or contact external services.

The fixed prompt fixture is `quality/diffusion_prompts_v0.tsv`. A diffusion serving claim must run all
30 prompts and report the `text_completion`, `math_reasoning`, and `instruction_following` slices
separately.

Future heavy evaluator command shape:

```bash
scripts/diffusion-eval/run.sh \
  --reference <hf-or-bf16-coreai-reference> \
  --candidate-bundle <diffusion-candidate> \
  --prompts quality/diffusion_prompts_v0.tsv \
  --out quality/raw/<run>
```

Real runs must write:

- `metadata.json`
- `diffusion_quality.tsv`
- `diffusion_api.json`
- `summary.json`

Use `scripts/diffusion-eval/validate-run.py --run quality/raw/<run>` after a heavy run to validate the
raw evidence shape before making any model-card, API, or catalog claim.

The validator keeps failed evidence usable when `summary.json` says `blocked`, but a
`final_decision` of `pass` must agree with the raw rows: all fixed-prompt rubric rows pass, per-slice
`passed`/`total` counts match `diffusion_quality.tsv`, and `diffusion_api.json` has `pass: true`.
