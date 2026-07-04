#!/usr/bin/env bash
# Keep the fast CoreAILM path off blanket fixed-size KV allocation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE="$REPO_DIR/Sources/PipelineRuntime/PipelinedLLM.swift"
VENDORED="$REPO_DIR/.build/checkouts/coreai-models/swift/Sources/CoreAILanguageModels/InferenceEngines/EngineFactory.swift"

[[ -f "$SOURCE" ]] || { echo "error: missing PipelinedLLM source: $SOURCE" >&2; exit 2; }
[[ -f "$VENDORED" ]] || { echo "error: missing vendored EngineFactory source: $VENDORED" >&2; exit 2; }

if rg -n 'kvCacheStrategy:[[:space:]]*\.fixedSize|KVCacheStrategy[.]fixedSize|[.]fixedSize' "$SOURCE"; then
  echo "error: PipelinedLLM must not use blanket fixed-size KV cache allocation" >&2
  echo "hint: use EngineOptions()/.auto or a reviewed request-bounded kvCacheSize policy" >&2
  exit 1
fi

if rg -n 'kvCacheSize:[[:space:]]*[^n]' "$SOURCE"; then
  echo "error: explicit PipelinedLLM kvCacheSize needs a reviewed request-bounded guardrail" >&2
  echo "hint: document prompt+max+headroom sizing and cap/recreate behavior before enabling it" >&2
  exit 1
fi

if ! rg -q 'EngineFactory[.]createEngine\(' "$SOURCE"; then
  echo "error: PipelinedLLM no longer calls EngineFactory.createEngine; review KV guardrail" >&2
  exit 1
fi

if ! rg -q 'Avoid `[.]fixedSize` unless you need a known upper bound' "$VENDORED"; then
  echo "error: upstream EngineOptions fixed-size warning moved; review PipelinedLLM KV guardrail" >&2
  exit 1
fi

echo "pipelined KV guardrail ok"
