#!/usr/bin/env bash
# Reject direct token transport in repo code and docs. Use local CLI auth instead.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-token-handling.sh [path ...]

Default paths:
  Sources scripts python docs web Formula README.md Tests Package.swift

Fails when repo code/docs reintroduce direct HF token env reads, Bearer auth headers,
or token argv. This does not inspect local auth files.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -gt 0 ]]; then
  paths=("$@")
else
  paths=(Sources scripts python docs web Formula README.md Tests Package.swift)
fi

existing=()
for path in "${paths[@]}"; do
  if [[ -e "$path" ]]; then
    existing+=("$path")
  fi
done

if [[ "${#existing[@]}" -eq 0 ]]; then
  echo "error: no input paths exist" >&2
  exit 2
fi

fail=0

scan() {
  local label="$1"
  local pattern="$2"
  if rg -n -i \
    --glob '!benchmarks/raw/**' \
    --glob '!scripts/check-token-handling.sh' \
    "$pattern" "${existing[@]}"; then
    echo "error: $label" >&2
    fail=1
  fi
}

scan "direct HF token env use" \
  '\b(HF_TOKEN|HUGGING_FACE_HUB_TOKEN)\b'
scan "Bearer Authorization header" \
  'Authorization[^[:cntrl:]]*Bearer|Bearer[^[:cntrl:]]*Authorization'
scan "token passed as argv" \
  '(^|[[:space:]])--token([[:space:]=]|$)'
scan "stale HF cache default; use /Volumes/SSD/hf-cache" \
  '<checkout-parent>/hf-cache|converter default is[^[:cntrl:]]*hf-cache|PIPELINE_ROOT[.]parent[[:space:]]*/[[:space:]]*"hf-cache"'
scan "stale converter tmp default; use /Volumes/SSD/coreai-tmp" \
  '<checkout-parent>/coreai-tmp|PIPELINE_ROOT[.]parent[[:space:]]*/[[:space:]]*"coreai-tmp"'

if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

echo "token handling ok"
