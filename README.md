# caix · native Apple Core AI inference server

[![Release](https://img.shields.io/github/v/release/RedHillsMediaFL/caix?include_prereleases&sort=semver&color=E5484D)](https://github.com/RedHillsMediaFL/caix/releases)
[![Stars](https://img.shields.io/github/stars/RedHillsMediaFL/caix)](https://github.com/RedHillsMediaFL/caix/stargazers)
[![Apple silicon · Core AI](https://img.shields.io/badge/Apple%20silicon-Core%20AI-black?logo=apple&logoColor=white)](https://github.com/RedHillsMediaFL/caix)
[![Swift](https://img.shields.io/badge/Swift-F05138?logo=swift&logoColor=white)](https://github.com/RedHillsMediaFL/caix)
[![🤗 Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-redhillsmediafl-ffcc4d)](https://huggingface.co/redhillsmediafl)

> **BETA.** caix is beta software and will remain so until Apple's **Core AI** runtime leaves
> beta. Things may change; expect rough edges. It runs entirely on your Mac — no data leaves the
> device.

**caix** runs large language models *natively* on Apple silicon through Apple's **Core AI** runtime
(Neural Engine + GPU), and serves them behind:

- a **web dashboard** (machine stats, model management, live usage),
- a dedicated **chat** view (full markdown, streaming, tools, skills, MCP),
- an **OpenAI-compatible** API (`/v1/chat/completions`) and **Anthropic-compatible** API (`/v1/messages`),

so any app that speaks the OpenAI or Anthropic API can use your local models.

It also includes **MTP / speculative decoding** (a small "draft" model proposes tokens that the big
model verifies in one pass — same output, faster) for supported model pairs.

---

## Performance

What local-model people care about: speed, size, runtime, and whether inference stays on the Mac.
caix runs on Apple's Core AI runtime and serves models through OpenAI/Anthropic-compatible APIs.

**Qwen3-4B**, ~4-bit, 200-token streaming decode, `temp=0`, Apple silicon 64 GB:

| engine | decode tok/s |
|---|---:|
| mlx | 121.7 |
| **caix** | **111.9** |
| llama.cpp | 90.8 |
| ollama | 56.5 |

Bottom line: caix does **111.9 tok/s**. mlx is a little faster. caix beats llama.cpp and ollama here.

**gpt-oss-20b** MoE: caix does **67.9 tok/s**. mlx leads at **87.0 tok/s**; llama.cpp is **81.5
tok/s**; ollama is **55.0 tok/s**.

Numbers vary by Mac and model. These are current `main` numbers, not unreleased server-optimization
claims.

---

## What you need (requirements)

| | |
|---|---|
| **Mac** | Apple silicon (M1 or newer). More unified memory = bigger models. A 26B 4-bit model needs ~20 GB free. |
| **macOS** | A version that ships Apple's **Core AI** runtime (currently a **beta** OS — Core AI is itself beta). |
| **Xcode** | A **Core AI–capable Xcode** (beta) providing the `swift` toolchain and the `CoreAI` framework. |
| **Disk** | A few GB for the build + the size of each model (e.g. ~13 GB for the 26B 4-bit bundle). |
| **Internet** | Only for the one-time build (fetches Swift packages) and downloading models. Inference is 100% offline. |

> **Why the beta requirement?** caix is a thin, fast native layer on top of Apple's Core AI. Core AI
> is in beta and only ships in recent OS/Xcode betas, so caix requires them too. When Core AI ships,
> caix will target the stable toolchain.

Everything else (the HTTP server, tokenizer, etc.) is fetched automatically by Swift Package Manager
— **no manual third-party installs.**

---

## Install

One-line source install (requires the Core AI-capable macOS/Xcode beta from Requirements):

```bash
curl -fsSL https://raw.githubusercontent.com/RedHillsMediaFL/caix/main/scripts/install.sh | bash
```

This clones/updates `~/caix`, builds the Core AI runtime binary, and links the `caix` launcher into
`~/.local/bin` when possible.

### Option A — download a prebuilt binary (easiest, no build) ⭐

1. Go to the [**Releases**](https://github.com/redhillsmediafl/caix/releases) page and download
   `caix-<version>-macos-arm64.tar.gz`.
2. Extract and run:

```bash
tar -xzf caix-*-macos-arm64.tar.gz
cd caix-*-macos-arm64
./caix serve
```

The first time you run a downloaded binary, macOS Gatekeeper may block it — allow it once:
**System Settings → Privacy & Security → "Open Anyway"**, or:

```bash
xattr -dr com.apple.quarantine ./bin/caix
```

> The prebuilt binary needs a Mac whose macOS includes the matching **Core AI** runtime (same beta
> line it was built against). If it won't launch (`CoreAI.framework` load error), use Option B.

### Option B — build from source

```bash
git clone https://github.com/redhillsmediafl/caix.git
cd caix
sudo xcode-select -s /Applications/Xcode-beta.app    # your Core AI–capable Xcode (once)
./scripts/install.sh                                 # = COREAI_RUNTIME=1 swift build -c release
```

The first build downloads the Swift dependencies and compiles — a few minutes. Dependencies are
fetched automatically; nothing to install by hand.

> **Pinned dependency (beta).** `Package.swift` pins `apple/coreai-models` to an exact revision
> known-good against the current Core AI beta. As Apple's betas move, that revision may need bumping
> (and the deployment target raised) to match your installed runtime — edit the `revision:` in
> `Package.swift`, or `swift package update`. We'll track the stable Core AI release when it lands.

---

## Get a model

caix serves **`.aimodel` bundles** (Core AI's format). Two ways to get one:

**A. Download a ready-made bundle (easiest).** Grab a pre-converted bundle from Hugging Face and put
it in `models/exports/`:

```bash
mkdir -p models/exports
# a small, capable model that's also a reliable tool-caller:
hf download redhillsmediafl/rhm-qwen3-4b-caix --local-dir models/exports/qwen3-4b-coreai
```

### Available models — org [redhillsmediafl](https://huggingface.co/redhillsmediafl)

All bundles are pre-converted Core AI `.aimodel` repos, named `rhm-…-caix`.

**Standalone models**

| model | base | size |
|---|---|---|
| [`rhm-qwen3-0.6b-caix`](https://huggingface.co/redhillsmediafl/rhm-qwen3-0.6b-caix) | Qwen3-0.6B — tiny & fast; handy edge/draft model | ~335 MB |
| [`rhm-qwen3-4b-caix`](https://huggingface.co/redhillsmediafl/rhm-qwen3-4b-caix) | Qwen3-4B — general chat, **reliable tool-caller** | ~2.1 GB |
| [`rhm-ornith-1.0-9b-caix`](https://huggingface.co/redhillsmediafl/rhm-ornith-1.0-9b-caix) | Ornith-1.0-9B (DeepReinforce) | ~17 GB (f16) |
| [`rhm-qwythos-9b-caix`](https://huggingface.co/redhillsmediafl/rhm-qwythos-9b-caix) | Qwythos-9B-Claude-Mythos-5-1M — qwen3_5 hybrid with visible-content template patch | ~4.7 GB |
| [`rhm-glm-4-9b-0414-caix`](https://huggingface.co/redhillsmediafl/rhm-glm-4-9b-0414-caix) | GLM-4-9B-0414 — dense bilingual model | ~5.0 GB |
| [`rhm-glm-4-32b-0414-caix`](https://huggingface.co/redhillsmediafl/rhm-glm-4-32b-0414-caix) | GLM-4-32B-0414 — dense large model | ~17 GB |
| [`rhm-gpt-oss-20b-caix`](https://huggingface.co/redhillsmediafl/rhm-gpt-oss-20b-caix) | OpenAI gpt-oss-20b — verified 4-bit Core AI bundle | ~10 GB |
| [`rhm-mixtral-8x7b-instruct-caix`](https://huggingface.co/redhillsmediafl/rhm-mixtral-8x7b-instruct-caix) | Mixtral-8x7B-Instruct-v0.1 — sparse MoE instruction model | ~24 GB |
| [`rhm-qwen3.6-27b-caix`](https://huggingface.co/redhillsmediafl/rhm-qwen3.6-27b-caix) | Qwen3.6-27B — large general chat | ~14 GB |
| [`rhm-gemma-4-26b-a4b-caix`](https://huggingface.co/redhillsmediafl/rhm-gemma-4-26b-a4b-caix) | gemma-4-26B-A4B-it — MoE (~4B active) | ~13 GB |
| [`rhm-gemma-4-31b-it-caix`](https://huggingface.co/redhillsmediafl/rhm-gemma-4-31b-it-caix) | gemma-4-31B-it — dense large chat | ~16 GB |

**MTP / speculative bundles** (draft + target in one repo — faster decode, identical output)

| model | what | size |
|---|---|---|
| [`rhm-gemma-4-26b-a4b-mtp-caix`](https://huggingface.co/redhillsmediafl/rhm-gemma-4-26b-a4b-mtp-caix) | gemma-4-26B-A4B **MTP** — fast flagship | ~17 GB |
| [`rhm-gemma-4-31b-it-mtp-caix`](https://huggingface.co/redhillsmediafl/rhm-gemma-4-31b-it-mtp-caix) | gemma-4-31B-it **MTP** | 31B target + draft |
| [`rhm-qwen3-4b-mtp-caix`](https://huggingface.co/redhillsmediafl/rhm-qwen3-4b-mtp-caix) | Qwen3-4B **MTP** | 4B target + 0.6B draft |

**Draft models** (EAGLE/MTP components; advanced — pair with the matching target)

| model | for |
|---|---|
| [`rhm-gemma-4-26b-a4b-draft-caix`](https://huggingface.co/redhillsmediafl/rhm-gemma-4-26b-a4b-draft-caix) | draft half of the 26B-A4B MTP pair |
| [`rhm-gemma-4-31b-it-draft-caix`](https://huggingface.co/redhillsmediafl/rhm-gemma-4-31b-it-draft-caix) | draft half of the 31B-it MTP pair |

An MTP repo is a **two-model** system (a target + a small draft); see its card for the exact
`--eagle-*` flags.

**B. Convert one yourself (advanced).** See [Converting your own models](#converting-your-own-models-advanced).

A bundle is just a folder containing a `*.aimodel/` directory, a `metadata.json`, and a `tokenizer/`
folder. Drop it under `models/exports/` and caix will list it.

---

## Run it

```bash
./caix serve
```

Then open:

- **http://localhost:1237** — the dashboard (machine stats, models, live usage)
- **http://localhost:1237/chat** — the chat view (markdown, streaming, tools, skills, MCP)

Use it from any OpenAI-compatible app by pointing it at `http://localhost:1237/v1` (any API key):

```bash
curl http://localhost:1237/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "<your-model-name>",
  "messages": [{"role":"user","content":"Hello!"}]
}'
```

### OpenCode (optional)

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

### Reach it from your phone / other machines (optional)
caix binds to `127.0.0.1` (local only) by design. To reach it securely from your other devices,
put it behind [Tailscale](https://tailscale.com): `tailscale serve --bg http://127.0.0.1:1237`
gives you a private HTTPS URL on your tailnet (no ports opened to the internet).

---

## Features

- **Native Core AI** inference on the Neural Engine + GPU (not llama.cpp / MLX — Apple's own runtime).
- **MTP / speculative decoding** for supported pairs — faster generation, identical output.
- **Dashboard** — live machine stats (RAM/GPU), model load/unload/convert/delete, **usage analytics**
  (total tokens in/out, requests, rolling + lifetime tok/s, **per-model** breakdown). Stats **persist
  across restarts**.
- **Chat view** — full markdown (code blocks with copy, tables), streaming, a "thinking" panel for
  reasoning models, plus:
  - **Skills** — pick or write a system prompt.
  - **Tools** — built-in `calculator`, `clock`, `fetch_url`; the model calls them and you see each
    call + result inline. *(Works best with qwen-family models; the fast gemma MTP model is a weak
    tool-caller.)*
  - **MCP** — connect a streamable-HTTP MCP server and use its tools in chat.
- **APIs** — OpenAI (`/v1/chat/completions`, `/v1/models`) and Anthropic (`/v1/messages`), streaming.
- **Model management from the UI** — download + convert from a Hugging Face repo, with new
  architectures flagged with the Core AI authoring work required.

---

## Beta quirks & known limitations

- **Beta toolchain required** (see Requirements). This is the big one.
- **First request after starting/loading a model is slow** — Core AI compiles the model on first use
  (~30–60 s for a 13 GB model). The chat view shows a "loading model…" spinner. Subsequent requests
  are fast.
- **Server fast path:** standard language bundles use Apple's CoreAILanguageModels one-shot
  generation path by default. Set `COREAI_PERSISTENT_FAST_ENGINE=1` to test the experimental
  kept-hot fast engine, or `COREAI_LEGACY_ENGINE=1` only when debugging the older sequential path.
  Hybrid qwen3_5/qwen3_5_moe bundles with packed recurrent state route to the explicit Core AI
  sequential engine unless `COREAI_FAST_HYBRID_ENGINE=1` is set, because the fast warmup cache shape
  is currently too small for those fixed state prefixes.
- **Background services & external volumes:** if you run caix from a `launchd` agent (auto-start),
  Apple's loader needs file-access permission and can't read **external/USB volumes** without it.
  Simplest: keep the binary + models on the internal disk, or run `./caix serve` from your normal
  user session. (Full Disk Access on the binary fixes the launchd case.) If you intentionally keep
  exports on an external SSD, generate a local model-list index and pass it via `CAIX_EXPORT_INDEX`:

```bash
scripts/refresh-export-index.sh /Volumes/SSD/ai-dev/coreai-pipeline/exports \
  ~/coreai-server/export-index.json
```
- **Greedy tool-calling:** the MTP/speculative path is greedy, so small models may not reliably emit
  tool calls. Use a qwen-family model for tool-heavy chats.
- **Authored architectures** (for conversion): gemma3/gemma4, qwen2/qwen3/qwen3_moe/qwen3_5/qwen3_5_moe,
  glm4, mistral, mixtral, gpt_oss. New model types are flagged with their required Core AI
  authoring steps in the UI and support logs.
- **Ornith frontier lane:** `ornith-1.0-35b` is registered for local conversion. The authored int4
  path now exports a 17 GB full bundle, but full runtime warmup is still quarantined behind an MPS
  reshape failure; it will not be published to HF until live smoke passes. Current depth probes pass
  generation through 13 layers and fail at the 14-layer decode shape, pointing the next support step
  toward graph segmentation or a decode-shape workaround. The 397B variant uses the same authored
  `qwen3_5_moe` path but requires its full 122-shard source download first.
- **Diffusion models** are on the roadmap, not in this beta.

---

## Converting your own models (advanced)

Conversion wraps Apple's `coreai.llm.export` and needs Apple's **coreai-models** Python checkout and
[`uv`](https://github.com/astral-sh/uv):

```bash
# point caix at your coreai-models python dir + (optionally) caches:
export CAIX_COREAI_MODELS=/path/to/coreai-models/python
export HF_HOME=~/.cache/huggingface
export CAIX_EXPORTS=$PWD/models/exports

# check support, then convert:
python3 python/converter/convert.py --check --hf-id Qwen/Qwen2-0.5B
python3 python/converter/convert.py --hf-id Qwen/Qwen2-0.5B --compression 4bit --compute-precision float16
```

Only run one heavy Core AI export or CLI verification at a time. For local queues, gate each job with:

```bash
scripts/conversion-guard.sh --wait
```

Or do it from the dashboard: paste a HF repo in **Add model → HuggingFace → Core AI**, pick
compression/precision, and click Convert. New architectures are flagged with the authoring work
needed before conversion.

### GGUF repos (llama.cpp models)

caix can also convert **GGUF** repos. If a repo ships only `.gguf` files (no safetensors), caix
**dequantizes** the GGUF back to an HF checkpoint, then runs the normal export — so you can pull a
llama.cpp model straight into a Core AI `.aimodel`:

```bash
# repo: caix picks the highest-quality quant present (F16 > Q8_0 > Q6_K > …)
python3 python/converter/convert.py --gguf unsloth/Qwen3-0.6B-GGUF --name qwen3-0.6b-coreai
# or pin a specific file:
python3 python/converter/convert.py --gguf unsloth/Qwen3-0.6B-GGUF --gguf-file Qwen3-0.6B-BF16.gguf
```

From the dashboard, just paste a GGUF-only repo — it's detected automatically and routed through the
dequant step (the support check confirms the architecture *after* dequant).

> **Quality caveat.** A GGUF is already quantized; dequantizing then re-exporting (often to 4-bit)
> is quant-on-quant and loses quality versus converting the **original safetensors**. Prefer the
> original release when it exists; use GGUF only when that's all that's published. The architecture
> still has to be one caix supports (below) — GGUF doesn't add new architectures.
> Needs `gguf>=0.10.0` (added automatically for the dequant run). Note: `gemma4` GGUFs are **not**
> convertible — `transformers` has no GGUF dequantizer for that architecture yet.

---

## Where things live

| | |
|---|---|
| Models | `models/exports/<name>/` (override with `CAIX_EXPORTS` or `--exports`) |
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
