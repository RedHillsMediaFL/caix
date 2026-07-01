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

Benchmark procedure is in [docs/BENCHMARKS.md](docs/BENCHMARKS.md) and [docs/TESTING.md](docs/TESTING.md).

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
brew install redhillsmediafl/caix/caix
caix doctor
```

After macOS 27/Core AI are stable and the release formula passes audit/test:

```bash
brew tap RedHillsMediaFL/caix
brew install caix
caix doctor
```

Homebrew/core comes later. See [Homebrew notes](docs/HOMEBREW.md).
Releases stay below `1.0.0` while Core AI is beta; see [release notes](docs/RELEASES.md).

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

**A. Download a converted bundle.** The default install path is `~/.caix/models/exports`:

```bash
# list converted Qwen repos and exact install commands
caix catalog redhillsmediafl/qwen
# prompt through families and models, then download:
caix catalog install
# or install one repo directly:
caix catalog install redhillsmediafl/rhm-qwen2.5-0.5b-instruct-caix
```

`caix catalog install` wraps `hf download`, uses local Hugging Face credentials/cache when present,
and runs a disk preflight first.

### Model catalog

Converted bundles are in the [redhillsmediafl Hugging Face org](https://huggingface.co/redhillsmediafl?search=caix)
and in family collections:

- [Qwen caix](https://huggingface.co/collections/redhillsmediafl/qwen-caix-6a3fd5f6d272c154dbfcda67)
- [Gemma caix](https://huggingface.co/collections/redhillsmediafl/gemma-caix-6a3fd5f57b67589b85e6eac6)
- [GLM caix](https://huggingface.co/collections/redhillsmediafl/glm-caix-6a40604dfb87315cc99a558e)
- [Mistral caix](https://huggingface.co/collections/redhillsmediafl/mistral-caix-6a404e43a972c2f2621a000e)
- [Ornith caix](https://huggingface.co/collections/redhillsmediafl/ornith-caix-6a3ff0de0d269f65f53ef064)
- [Qwythos caix](https://huggingface.co/collections/redhillsmediafl/qwythos-caix-6a409e86fb87315cc9a2d69f)
- [gpt-oss caix](https://huggingface.co/collections/redhillsmediafl/gpt-oss-caix-6a404e428791003d8e6e79fc)

Use the catalog for current install commands:

```bash
caix catalog redhillsmediafl
caix catalog redhillsmediafl/qwen
caix catalog install <repo>
```

Model cards list exact revisions, licenses, storage, RAM notes, and any speculative or EAGLE flags.

**B. Convert a bundle.** See [Converting models](#converting-models-advanced).

A bundle is a folder containing a `*.aimodel/` directory, `metadata.json`, and `tokenizer/`. Drop it
under `models/exports/` and caix will list it.

---

## Run

```bash
caix serve
caix dashboard
caix chat
```

From a source checkout, `./caix serve` also works.

Open:

- **http://localhost:1237** — the dashboard (machine stats, models, live usage)
- **http://localhost:1237/chat** — the chat view (markdown, streaming, tools, skills, MCP)

`caix serve` prewarms the smallest chat-suitable installed model before it starts listening. Use
`--prewarm <model|all|off>` or `--no-prewarm` to override.

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
installer or `caix catalog install` to add the converted bundle first.

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
- MTP/speculative decoding for supported pairs; caix auto-tunes the draft window up to the safe cap.
- Dashboard: RAM/GPU stats, load/unload/convert/delete, per-model usage, rolling/lifetime tok/s.
  Usage stats persist across restarts.
- Chat: markdown, streaming, code blocks with copy, tables, and a thinking panel for reasoning
  models. The chat page includes a redacted session log for debugging.
- Terminal chat: `caix chat` / `caix tui` talks to a local caix server and includes shell-tool
  controls (`ask`, `on`, `off`).
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
- **Cold loads are slow.** Core AI compiles the model before first use (~30-60 s for a 13 GB model).
  `caix serve` prewarms by default; `--no-prewarm` accepts traffic immediately.
- **Server fast path.** Standard language bundles use Apple's kept-hot CoreAILanguageModels engine
  by default after `/api/load`. `COREAI_LEGACY_ENGINE=1` uses the older sequential path for
  debugging. Hybrid qwen3_5/qwen3_5_moe bundles with packed recurrent state use the explicit Core
  AI sequential engine unless `COREAI_FAST_HYBRID_ENGINE=1` is set; the current fast warmup cache
  shape is too small for those fixed state prefixes.
- **Background services and external volumes.** If you run caix from a `launchd` agent, Apple's
  loader needs file-access permission and cannot read external/USB volumes without it. Keep the
  binary and models on the internal disk, or run `caix serve` from a normal user session. Full Disk
  Access on the binary fixes the `launchd` case. If exports stay on an external SSD, generate a
  local model-list index and pass it via `caix_export_index`:

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
export caix_coreai_models=/path/to/coreai-models/python
export HF_HOME=${HF_HOME:-/Volumes/SSD/hf-cache}
export caix_tmpdir=${caix_tmpdir:-/Volumes/SSD/coreai-tmp}
export caix_exports=${caix_exports:-$PWD/models/exports}
scripts/check-disk-pressure.sh --path /Volumes/SSD --floor-gib 500

# check support, then convert:
python3 python/converter/convert.py --check --hf-id Qwen/Qwen2-0.5B
python3 python/converter/convert.py --hf-id Qwen/Qwen2-0.5B --compression 4bit --compute-precision float16
```

Run only one heavy Core AI export, HF transfer, CLI verification, build, or benchmark at a time.
For local queues, gate each job with:

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
| Models | `~/.caix/models/exports/<name>/` (override with `caix_exports` or `--exports`) |
| HF cache | `$HF_HOME`; otherwise Hugging Face uses its local default |
| Converter tmp | `$caix_tmpdir`; converter default is `/Volumes/SSD/coreai-tmp` |
| Export index | optional JSON from `scripts/refresh-export-index.sh` (set `caix_export_index`) |
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

<sub>More open-source work: [redhillsmediafl.com/open-source](https://redhillsmediafl.com/open-source).</sub>
