#!/usr/bin/env bash
# Run a gated real-model structured-output smoke against a CoreAILM language bundle.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: CAIX_STRUCTURED_OUTPUT_MODEL=/path/to/bundle scripts/check-structured-output-smoke.sh [--output <path>]

This is a model-loading smoke, not a no-load publication gate. It proves the CoreAILM
constrained-decoding path can emit JSON matching a tiny schema for the supplied bundle.
USAGE
}

OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUT="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) echo "error: unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

MODEL=${CAIX_STRUCTURED_OUTPUT_MODEL:?set CAIX_STRUCTURED_OUTPUT_MODEL to a CoreAILM language bundle}
if [[ ! -d "$MODEL" ]]; then
  echo "error: model bundle does not exist: $MODEL" >&2
  exit 1
fi

if [[ -n "$OUT" ]]; then
  mkdir -p "$(dirname "$OUT")"
  export CAIX_STRUCTURED_OUTPUT_SMOKE_OUTPUT="$OUT"
fi

if [[ -n "$OUT" ]]; then
  LOG="${OUT%.*}.log"
  if ! COREAI_RUNTIME=1 swift test -c release \
      --filter StructuredOutputSmokeTests/testRealCoreAILMStructuredOutputSmoke \
      >"$LOG" 2>&1
  then
    cat "$LOG" >&2
    echo "structured output smoke failed; log=$LOG" >&2
    exit 1
  fi
else
  COREAI_RUNTIME=1 swift test -c release \
    --filter StructuredOutputSmokeTests/testRealCoreAILMStructuredOutputSmoke
fi

echo "structured output smoke ok"
