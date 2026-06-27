# caix · native Apple Core AI inference server

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
pip install -U huggingface_hub
# a small, capable model that's also a reliable tool-caller:
huggingface-cli download redhillsmediafl/qwen3-4b-caix --local-dir models/exports/qwen3-4b-coreai
```

### Available models — org [redhillsmediafl](https://huggingface.co/redhillsmediafl)

| model | what | size |
|---|---|---|
| [`redhillsmediafl/qwen3-4b-caix`](https://huggingface.co/redhillsmediafl/qwen3-4b-caix) | Qwen3-4B — general chat, reliable tool-caller | ~2 GB |
| [`redhillsmediafl/gemma-4-26b-a4b-mtp-caix`](https://huggingface.co/redhillsmediafl/gemma-4-26b-a4b-mtp-caix) | gemma-4-26B-A4B **MTP** (target + draft) — fast flagship | ~17 GB |

The MTP repo is a **two-model** system (a 26B target + a small draft); see its card for the exact
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
- **Model management from the UI** — download + convert from a Hugging Face repo (unsupported
  architectures are flagged), load, unload, delete.

---

## Beta quirks & known limitations

- **Beta toolchain required** (see Requirements). This is the big one.
- **First request after starting/loading a model is slow** — Core AI compiles the model on first use
  (~30–60 s for a 13 GB model). The chat view shows a "loading model…" spinner. Subsequent requests
  are fast.
- **Background services & external volumes:** if you run caix from a `launchd` agent (auto-start),
  Apple's loader needs file-access permission and can't read **external/USB volumes** without it.
  Simplest: keep the binary + models on the internal disk, or run `./caix serve` from your normal
  user session. (Full Disk Access on the binary fixes the launchd case.)
- **Greedy tool-calling:** the MTP/speculative path is greedy, so small models may not reliably emit
  tool calls. Use a qwen-family model for tool-heavy chats.
- **Supported architectures** (for conversion): gemma3/gemma4, qwen2/qwen3/qwen3_moe/qwen3_5,
  mistral, mixtral, gpt_oss. Others are flagged "unsupported" in the UI.
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

Or do it from the dashboard: paste a HF repo in **Add model → HuggingFace → Core AI**, pick
compression/precision, and click Convert. Unsupported architectures are flagged with a reason.

---

## Where things live

| | |
|---|---|
| Models | `models/exports/<name>/` (override with `CAIX_EXPORTS` or `--exports`) |
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
