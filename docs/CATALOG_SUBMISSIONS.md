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

Attach raw logs for failures and benchmarks. Speed claims need the raw benchmark directory, exact
model repo revision, caix commit, prompt, token budget, temperature, warmup count, and measured runs.

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
