# caix

[![Release](https://img.shields.io/github/v/release/RedHillsMediaFL/caix?include_prereleases&sort=semver&color=E5484D)](https://github.com/RedHillsMediaFL/caix/releases)
[![Apple silicon · Core AI](https://img.shields.io/badge/Apple%20silicon-Core%20AI-black?logo=apple&logoColor=white)](https://github.com/RedHillsMediaFL/caix)
[![Swift](https://img.shields.io/badge/Swift-F05138?logo=swift&logoColor=white)](https://github.com/RedHillsMediaFL/caix)
[![HF models](https://img.shields.io/badge/HF%20models-redhillsmediafl-ffcc4d)](https://huggingface.co/redhillsmediafl)

Local Core AI inference server for Apple silicon — private, on-device LLMs behind OpenAI- and
Anthropic-compatible APIs. Free and open, no paywall.

> **Beta.** caix depends on Apple's beta **Core AI** runtime. Expect breakage across macOS/Xcode
> beta lines. Inference is local; downloads still use the network.

caix serves Core AI `.aimodel` language bundles on Apple silicon (Neural Engine + GPU). It exposes:

- a **web dashboard** for machine stats, model management, and live usage,
- a **chat** view with markdown, streaming, tools, skills, and MCP,
- **OpenAI-compatible** (`/v1/chat/completions`) and **Anthropic-compatible** (`/v1/messages`) APIs.

OpenAI and Anthropic clients can point at the local server.

caix supports classic speculative packages and EAGLE target+draft packages. Use the model-card flags
for each package.

Multimodal status: caix parses OpenAI and Anthropic image/audio/video content blocks, but no verified
runtime bundle consumes them yet. Multimodal requests return a clear 400 until that changes.

---

## Testing and support

caix is free and open. No paywall.

Use [docs/TESTER_REQUESTS.md](docs/TESTER_REQUESTS.md) for current test requests. Keep the model
repo revision, prompt, token budget, temperature, warmup count, measured run count, and chat-template
mode unchanged. Send the raw benchmark directory.

Support conversions, benchmark time, hosting, and tester access:
[redhillsmediafl.com/open-source](https://redhillsmediafl.com/open-source).

Support is optional and never gates features.

---

## Performance

No public speed table is current.

Benchmark rules and raw-log requirements are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Do not
publish speed numbers without the raw run directory, exact model repo revision, caix commit,
hardware, OS build, and exact command.

Planned work is tracked in [docs/ROADMAP.md](docs/ROADMAP.md).
The dry-run staged-execution planner surface is documented in [docs/CLUSTER.md](docs/CLUSTER.md).

---

## Requirements

| | |
|---|---|
| **Mac** | Apple silicon (M1 or newer). Unified memory limits model size. A 26B 4-bit model needs ~20 GB free. |
| **macOS** | macOS 27 beta, or another beta line that ships Apple's Core AI runtime. |
| **Xcode** | Matching Core AI-capable Xcode beta with `swift` and `CoreAI.framework`. |
| **Disk** | A few GB for the build + the size of each model (e.g. ~13 GB for the 26B 4-bit bundle). |
| **Internet** | The build fetches Swift packages. Model downloads use the network. Inference is offline. |

caix links directly against Core AI. The OS, Xcode, and `apple/coreai-models` revision need to match
the active beta line. Stable toolchain support waits on stable Core AI.

Swift Package Manager fetches the HTTP server, tokenizer, and other dependencies. No manual
third-party installs.

---

## Install

### Option A — Homebrew

Homebrew is the intended default install path after tap testing and Core AI stabilization.

Current beta path:

```bash
brew tap RedHillsMediaFL/caix
brew install --HEAD caix
caix doctor
```

After macOS 27/Core AI are stable and the release formula passes audit/test:

```bash
brew tap RedHillsMediaFL/caix
brew install caix
caix doctor
```

Homebrew/core comes later. See [Homebrew notes](docs/HOMEBREW.md).

### Option B — source install

Requires the Core AI-capable macOS/Xcode beta from [Requirements](#requirements):

```bash
curl -fsSL https://raw.githubusercontent.com/RedHillsMediaFL/caix/main/scripts/install.sh | bash
```

This clones or updates `~/caix`, builds the Core AI runtime binary, and links the `caix` launcher into
`~/.local/bin` when possible.

### Option C — download a prebuilt binary

1. Go to the [**Releases**](https://github.com/redhillsmediafl/caix/releases) page and download
   `caix-<version>-macos-arm64.tar.gz`.
2. Extract and run:

```bash
tar -xzf caix-*-macos-arm64.tar.gz
cd caix-*-macos-arm64
./caix serve
```

The first downloaded run may be blocked by macOS Gatekeeper. Allow it once:
**System Settings → Privacy & Security → "Open Anyway"**, or:

```bash
xattr -dr com.apple.quarantine ./bin/caix
```

> The prebuilt binary needs a Mac whose macOS includes the matching **Core AI** runtime (same beta
> line it was built against). If it won't launch (`CoreAI.framework` load error), use Option D.

### Option D — build from source

```bash
git clone https://github.com/redhillsmediafl/caix.git
cd caix
sudo xcode-select -s /Applications/Xcode-beta.app    # your Core AI–capable Xcode (once)
./scripts/install.sh                                 # = COREAI_RUNTIME=1 swift build -c release
```

The first build downloads Swift dependencies and compiles. It takes a few minutes.

> **Pinned dependency (beta).** `Package.swift` pins `apple/coreai-models` to an exact revision
> tested against the current Core AI beta. As Apple's betas move, that revision may need bumping
> and the deployment target may need raising. Edit `revision:` in `Package.swift`, or run
> `swift package update`.

---

## Get a model

caix serves Core AI `.aimodel` bundles. Get one of these two ways:

**A. Download a converted bundle.** Put it in `models/exports/`:

```bash
mkdir -p models/exports
# list converted Qwen repos and exact install commands
caix catalog redhillsmediafl/qwen
# Qwen3-4B model with tool-calling support:
hf download redhillsmediafl/rhm-qwen3-4b-caix --revision <catalog-revision> --local-dir models/exports/qwen3-4b-coreai
```

### Available models — org [redhillsmediafl](https://huggingface.co/redhillsmediafl)

These are pre-converted Core AI `.aimodel` repos, named `rhm-...-caix`.

**Standalone models**

| model | base | size |
|---|---|---|
| [`rhm-qwen2.5-0.5b-instruct-caix`](https://huggingface.co/redhillsmediafl/rhm-qwen2.5-0.5b-instruct-caix) | Qwen2.5-0.5B-Instruct — verified | ~290 MB |
| [`rhm-qwen2.5-3b-instruct-caix`](https://huggingface.co/redhillsmediafl/rhm-qwen2.5-3b-instruct-caix) | Qwen2.5-3B-Instruct — verified; Qwen Research License | ~1.7 GB |
| [`rhm-qwen3-0.6b-caix`](https://huggingface.co/redhillsmediafl/rhm-qwen3-0.6b-caix) | Qwen3-0.6B — edge/draft model | ~335 MB |
| [`rhm-qwen3-4b-caix`](https://huggingface.co/redhillsmediafl/rhm-qwen3-4b-caix) | Qwen3-4B — general chat; tool calling | ~2.1 GB |
| [`rhm-qwen3-8b-caix`](https://huggingface.co/redhillsmediafl/rhm-qwen3-8b-caix) | Qwen3-8B — verified | ~4.3 GB |
| [`rhm-qwen3-14b-caix`](https://huggingface.co/redhillsmediafl/rhm-qwen3-14b-caix) | Qwen3-14B — verified | ~7.8 GB |
| [`rhm-ornith-1.0-9b-caix`](https://huggingface.co/redhillsmediafl/rhm-ornith-1.0-9b-caix) | Ornith-1.0-9B (DeepReinforce) | ~17 GB (f16) |
| [`rhm-qwythos-9b-caix`](https://huggingface.co/redhillsmediafl/rhm-qwythos-9b-caix) | Qwythos-9B-Claude-Mythos-5-1M — qwen3_5 hybrid with visible-content template patch | ~4.7 GB |
| [`rhm-glm-4-9b-0414-caix`](https://huggingface.co/redhillsmediafl/rhm-glm-4-9b-0414-caix) | GLM-4-9B-0414 — dense bilingual model | ~5.0 GB |
| [`rhm-glm-4-32b-0414-caix`](https://huggingface.co/redhillsmediafl/rhm-glm-4-32b-0414-caix) | GLM-4-32B-0414 — dense chat model | ~17 GB |
| [`rhm-gpt-oss-20b-caix`](https://huggingface.co/redhillsmediafl/rhm-gpt-oss-20b-caix) | OpenAI gpt-oss-20b — verified 4-bit Core AI bundle | ~10 GB |
| [`rhm-mistral-7b-instruct-v0.3-caix`](https://huggingface.co/redhillsmediafl/rhm-mistral-7b-instruct-v0.3-caix) | Mistral-7B-Instruct-v0.3 — verified | ~3.8 GB |
| [`rhm-mistral-nemo-instruct-2407-caix`](https://huggingface.co/redhillsmediafl/rhm-mistral-nemo-instruct-2407-caix) | Mistral-Nemo-Instruct-2407 — verified | ~6.4 GB |
| [`rhm-mistral-small-instruct-2409-caix`](https://huggingface.co/redhillsmediafl/rhm-mistral-small-instruct-2409-caix) | Mistral-Small-Instruct-2409 — verified; Mistral Research License | ~12 GB |
| [`rhm-mixtral-8x7b-instruct-caix`](https://huggingface.co/redhillsmediafl/rhm-mixtral-8x7b-instruct-caix) | Mixtral-8x7B-Instruct-v0.1 — sparse MoE instruction model | ~24 GB |
| [`rhm-qwen3.6-27b-caix`](https://huggingface.co/redhillsmediafl/rhm-qwen3.6-27b-caix) | Qwen3.6-27B — general chat | ~14 GB |
| [`rhm-gemma-4-26b-a4b-caix`](https://huggingface.co/redhillsmediafl/rhm-gemma-4-26b-a4b-caix) | gemma-4-26B-A4B-it — MoE (~4B active) | ~13 GB |
| [`rhm-gemma-4-31b-it-caix`](https://huggingface.co/redhillsmediafl/rhm-gemma-4-31b-it-caix) | gemma-4-31B-it — dense chat | ~16 GB |

**MTP / speculative bundles** (draft + target in one repo)

| model | what | size |
|---|---|---|
| [`rhm-gemma-4-26b-a4b-mtp-caix`](https://huggingface.co/redhillsmediafl/rhm-gemma-4-26b-a4b-mtp-caix) | gemma-4-26B-A4B **MTP** target + draft | ~17 GB |
| [`rhm-gemma-4-31b-it-mtp-caix`](https://huggingface.co/redhillsmediafl/rhm-gemma-4-31b-it-mtp-caix) | gemma-4-31B-it **MTP** | 31B target + draft |
| [`rhm-qwen3-4b-mtp-caix`](https://huggingface.co/redhillsmediafl/rhm-qwen3-4b-mtp-caix) | Qwen3-4B **MTP** | 4B target + 0.6B draft |

**Draft models** (EAGLE/MTP components; advanced — pair with the matching target)

| model | for |
|---|---|
| [`rhm-gemma-4-26b-a4b-draft-caix`](https://huggingface.co/redhillsmediafl/rhm-gemma-4-26b-a4b-draft-caix) | draft half of the 26B-A4B MTP pair |
| [`rhm-gemma-4-31b-it-draft-caix`](https://huggingface.co/redhillsmediafl/rhm-gemma-4-31b-it-draft-caix) | draft half of the 31B-it MTP pair |

An MTP repo is a two-model system: target plus draft. See its model card for exact speculative or
EAGLE flags.

External verification and benchmark submissions: [docs/TESTING.md](docs/TESTING.md).
Current tester request sheet: [docs/TESTER_REQUESTS.md](docs/TESTER_REQUESTS.md).

**B. Convert a bundle.** See [Converting models](#converting-models-advanced).

A bundle is a folder containing a `*.aimodel/` directory, `metadata.json`, and `tokenizer/`. Drop it
under `models/exports/` and caix will list it.

---

## Run

```bash
caix serve
```

From a source checkout, `./caix serve` also works.

Open:

- **http://localhost:1237** — the dashboard (machine stats, models, live usage)
- **http://localhost:1237/chat** — the chat view (markdown, streaming, tools, skills, MCP)

For OpenAI-compatible clients, use `http://localhost:1237/v1` with any API key:

```bash
curl http://localhost:1237/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "<your-model-name>",
  "messages": [{"role":"user","content":"Hello!"}]
}'
```

### OpenCode

The repo includes `opencode.json` with a local `caix` OpenAI-compatible provider pointed at
`http://127.0.0.1:1237/v1` and known RHM/caix model IDs. Copy or symlink it into
`~/.config/opencode/opencode.json`, start caix, then verify OpenCode can see the provider:

```bash
opencode models caix
```

OpenCode reads the static provider map; the live server list is the source of truth for installed
bundles:

```bash
curl http://127.0.0.1:1237/v1/models
```

When OpenCode sends an installed model ID to `/v1/chat/completions`, caix hot-loads the matching
local bundle from the exports directory. If the ID is not installed yet, use the dashboard's RHM
installer or `hf download` to add the converted bundle first.

### Other devices

caix binds to `127.0.0.1` by default. To reach it from other devices, put it behind
[Tailscale](https://tailscale.com):

```bash
tailscale serve --bg http://127.0.0.1:1237
```

This gives you a private HTTPS URL on your tailnet without opening public ports.

---

## What is included

- Core AI inference on Neural Engine + GPU. This is not llama.cpp or MLX.
- MTP/speculative decoding for supported pairs.
- Dashboard: RAM/GPU stats, load/unload/convert/delete, per-model usage, rolling/lifetime tok/s.
  Usage stats persist across restarts.
- Chat: markdown, streaming, code blocks with copy, tables, and a thinking panel for reasoning
  models.
- Skills: choose or write a system prompt.
- Tools: built-in `calculator`, `clock`, and `fetch_url`. Use qwen-family models for tool-heavy
  chats; the gemma MTP model is weak at tool calls.
- MCP: connect a streamable-HTTP MCP server and use its tools in chat.
- APIs: OpenAI (`/v1/chat/completions`, `/v1/models`) and Anthropic (`/v1/messages`), streaming.
- UI model management: download and convert from a Hugging Face repo. New architectures are flagged
  with the Core AI authoring work they need.

---

## Known limitations

- **Beta toolchain required.** See [Requirements](#requirements).
- **Cold loads are slow.** Core AI compiles the model on first use (~30-60 s for a 13 GB model).
  The chat view shows a "loading model..." spinner. Later requests reuse compiled state.
- **Server fast path.** Standard language bundles use Apple's kept-hot CoreAILanguageModels engine
  by default after `/api/load`. `COREAI_LEGACY_ENGINE=1` uses the older sequential path for
  debugging. Hybrid qwen3_5/qwen3_5_moe bundles with packed recurrent state use the explicit Core
  AI sequential engine unless `COREAI_FAST_HYBRID_ENGINE=1` is set; the current fast warmup cache
  shape is too small for those fixed state prefixes.
- **Background services and external volumes.** If you run caix from a `launchd` agent, Apple's
  loader needs file-access permission and cannot read external/USB volumes without it. Keep the
  binary and models on the internal disk, or run `caix serve` from a normal user session. Full Disk
  Access on the binary fixes the `launchd` case. If exports stay on an external SSD, generate a
  local model-list index and pass it via `CAIX_EXPORT_INDEX`:

```bash
scripts/refresh-export-index.sh /Volumes/SSD/ai-dev/coreai-pipeline/exports \
  ~/coreai-server/export-index.json
```

- **Greedy tool-calling.** The MTP/speculative path is greedy, so lower-parameter models may not
  reliably emit tool calls. Use a qwen-family model for tool-heavy chats.
- **Authored architectures.** Conversion support exists for gemma3/gemma4,
  qwen2/qwen3/qwen3_moe/qwen3_5/qwen3_5_moe, glm4, mistral, mixtral, and gpt_oss. New model types
  are flagged with their required Core AI authoring steps in the UI and support logs.
- **Ornith-1.0-35B lane.** `ornith-1.0-35b` is registered for local conversion. The authored int4
  path exports a 17 GB bundle, but runtime warmup is blocked by an MPS reshape failure, so it is not
  published to HF. The next fix is graph segmentation or a decode-shape workaround. The 397B variant
  uses the same authored `qwen3_5_moe` path and needs the full 122-shard source download first.
- **Diffusion models** are not in this beta.

---

## Converting models (advanced)

Conversion wraps Apple's `coreai.llm.export` and needs Apple's `coreai-models` Python checkout and
[`uv`](https://github.com/astral-sh/uv):

```bash
# point caix at your coreai-models python dir and keep conversion IO on the SSD:
export CAIX_COREAI_MODELS=/path/to/coreai-models/python
export HF_HOME=${HF_HOME:-/Volumes/SSD/hf-cache}
export CAIX_TMPDIR=${CAIX_TMPDIR:-/Volumes/SSD/coreai-tmp}
export CAIX_EXPORTS=${CAIX_EXPORTS:-$PWD/models/exports}
scripts/check-disk-pressure.sh --path /Volumes/SSD --floor-gib 500

# check support, then convert:
python3 python/converter/convert.py --check --hf-id Qwen/Qwen2-0.5B
python3 python/converter/convert.py --hf-id Qwen/Qwen2-0.5B --compression 4bit --compute-precision float16
```

Run only one heavy Core AI export or CLI verification at a time. For local queues, gate each job with:

```bash
scripts/conversion-guard.sh --wait
```

Dashboard path: paste a HF repo in **Add model → HuggingFace → Core AI**, pick
compression/precision, and click Convert. New architectures are flagged with the required authoring
work.

### GGUF repos (llama.cpp models)

caix can convert GGUF repos. If a repo ships only `.gguf` files, caix dequantizes the GGUF back to
an HF checkpoint, then runs the normal export.

```bash
# repo: caix picks the least-compressed quant present (F16 > Q8_0 > Q6_K > ...)
python3 python/converter/convert.py --gguf unsloth/Qwen3-0.6B-GGUF --name qwen3-0.6b-coreai
# or pin a specific file:
python3 python/converter/convert.py --gguf unsloth/Qwen3-0.6B-GGUF --gguf-file Qwen3-0.6B-BF16.gguf
```

Dashboard path: paste a GGUF-only repo. It is routed through the dequant step; the support check
confirms the architecture after dequant.

> **Quality caveat.** A GGUF is already quantized. Dequantizing and re-exporting, often to 4-bit, is
> quant-on-quant and loses quality versus converting the original safetensors. Prefer the original
> release when it exists. GGUF does not add architecture support.
> Needs `gguf>=0.10.0` (added automatically for the dequant run). Note: `gemma4` GGUFs are **not**
> convertible — `transformers` has no GGUF dequantizer for that architecture yet.

---

## Paths

| | |
|---|---|
| Models | `models/exports/<name>/` (override with `CAIX_EXPORTS` or `--exports`) |
| HF cache | `$HF_HOME`; caix defaults it to `/Volumes/SSD/hf-cache` when unset |
| Converter tmp | `$CAIX_TMPDIR`; converter default is `<checkout-parent>/coreai-tmp` |
| Export index | optional JSON from `scripts/refresh-export-index.sh` (set `CAIX_EXPORT_INDEX`) |
| Web UI | `web/` (served at `/` and `/chat`) |
| Usage stats | `~/.caix/usage.json` (override with `--stats-file`) — survives restarts |
| Converter | `python/converter/` |

---

## Troubleshooting

- **`swift: command not found`** → install Xcode (beta) and `sudo xcode-select -s /Applications/Xcode-beta.app`.
- **Build fails fetching `coreai-models`** → that package requires your Core AI–capable Xcode; confirm
  `xcode-select -p` points at it.
- **`/v1` returns 503** → the binary was built **without** the runtime. Rebuild with
  `COREAI_RUNTIME=1 swift build -c release` (the `./caix` launcher and `install.sh` do this for you).
- **No models listed** → put a bundle under `models/exports/` (a folder with a `*.aimodel/` inside).
- **First message hangs ~1 min** → expected on cold load; watch the spinner.

---

## License

See [LICENSE](LICENSE). caix bundles `marked` and `DOMPurify` (their licenses in `web/assets/`).

---

*caix is beta and unaffiliated with Apple. "Core AI" is Apple's runtime.*
