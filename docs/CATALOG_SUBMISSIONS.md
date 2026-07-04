# Catalog Submissions

caix catalog entries are for open-source or open-weight models with redistribution terms we can state
on the model card. Do not submit closed-weight, unclear-license, or private-token-dependent payloads.

## Publication Gates

A model can go in the catalog only as one of these states:

| state | required evidence |
|---|---|
| `verified` | `caix inspect` plus load/generation smoke on target hardware; benchmark rows need raw logs |
| `needs-test` | structural checks passed, but the needed hardware is unavailable or lacks enough unified memory |
| `component` | draft/assistant/MTP component; card names the matching target package |
| `blocked` | converted or attempted, but runtime load/generation fails; not a normal install target |

For `needs-test`, state the exact gap: memory ceiling, missing second machine, missing Thunderbolt
test, or Core AI/runtime issue. Do not imply runtime success.

Every public RHM card must satisfy the card-v2 metadata contract before catalog promotion:
`library_name: caix`, `## Download`, `## License`, the support link, and evidence rows for Base
model, Format, Quant, Context, Runtime, and License. Component, staged, blocked, and
instability-gated cards must include a `caix-status-label` block and must not claim ready-to-run or
verified status. A card that does claim ready-to-run or verified-in-caix status must cite parity
evidence and speed/benchmark evidence.
MTP/speculative package cards must also include a `caix-status-label` block that identifies the
target+draft package and names the matching standalone target context; publish target-only and
target+draft results separately.
Use `<model>-staged` for the artifact that can run all stages locally or split them across workers,
and `<model>-monolithic` for a separate single-machine fused fast path. A monolithic card needs a
determinism/parity status block. The reviewed qwen3-4b monolithic fix makes stateful prefill
deterministic by default with a `<=16` chunk cap, but it remains a 4bit path and must not claim
fp16 1:1; keep it unpublished/unrelabelled until the fix is accepted and release-gated.
`scripts/check-publication-gates.sh --hub` enforces this through `scripts/check-hf-model-cards.sh`
without downloading model payloads.

## Submit a Model

Open a GitHub issue with:

| field | value |
|---|---|
| source repo | Hugging Face repo id and revision |
| license | source license and redistribution note |
| architecture | model type, layers, hidden size, context, quantization |
| artifact | `.aimodel` layout, `metadata.json`, tokenizer files, and README |
| caix version | `caix --version` |
| verification | exact commands and pass/fail output |
| hardware | chip, unified memory, macOS build |

For staged/distributed packages, include:

```bash
caix cluster plan --manifest <bundle>/stage-manifest.json --workers studio=64,macbook=32 --kv-capacity 128
caix deploy verify --endpoint <host-a>:1237 --endpoint <host-b>:1237 --min-mbps <floor>
```

If the second machine is unavailable, keep the staged package in `needs-test`; do not promote it
from Studio-only `cluster plan`, loopback/socket smoke, or diagnostic HF parity evidence.

Attach raw logs for failures and benchmarks. Speed claims need the raw benchmark directory, exact
model repo revision, caix commit, prompt, token budget, temperature, warmup count, and measured runs.
Release and capability claims also need `benchmarks/DEPENDENCY_EVIDENCE.tsv` to match the current
resolved Core AI dependency set.
Quality claims need the relevant gate evidence from [QUALITY_GATES.md](QUALITY_GATES.md), including
raw `quality/raw/<run>/` artifacts for the exact model revision.
Diffusion cards also need the API contract state from [DIFFUSION_API.md](DIFFUSION_API.md): v1 is
non-streaming-only, and committed-block SSE is not a supported claim until separately implemented and
tested.
Diffusion quality evidence must also validate against `quality/diffusion_prompts_v0.tsv` and include
`quality/raw/<run>/diffusion_quality.tsv`.

## Submit Test Results

Install with the catalog so the same revision is tested:

```bash
caix catalog install <repo> --revision <revision>
caix inspect --model ~/.caix/models/exports/<name>
caix run --model ~/.caix/models/exports/<name> --prompt "Name one primary color." --max-tokens 32 --temperature 0 --verbose
```

For distributed packages, test installed `caix` from Homebrew and include `caix deploy verify`
output before the model smoke.

Do not include tokens, auth headers, private URLs, or local credential files.

<sub>More open-source work: [redhillsmediafl.com/open-source](https://redhillsmediafl.com/open-source).</sub>
