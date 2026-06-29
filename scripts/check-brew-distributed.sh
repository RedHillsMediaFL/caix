#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-brew-distributed.sh [--caix <path>] [--ready] [--manifest <path>]

Checks the Homebrew-installed caix surface needed before Thunderbolt distributed tests.
It does not start workers or run inference.
USAGE
}

caix_binary="${caix_bin:-caix}"
require_ready=0
manifest=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --caix) caix_binary="${2:?}"; shift 2 ;;
    --ready) require_ready=1; shift ;;
    --manifest) manifest="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) echo "error: unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$caix_binary" == */* ]]; then
  [[ -x "$caix_binary" ]] || {
    echo "error: caix binary not found or not executable: $caix_binary" >&2
    exit 1
  }
else
  caix_binary="$(command -v "$caix_binary")" || {
    echo "error: caix binary not found: $caix_binary" >&2
    exit 1
  }
fi

"$caix_binary" --version
"$caix_binary" doctor --no-fail
"$caix_binary" cluster plan --help >/dev/null

if [[ -n "$manifest" ]]; then
  json="$("$caix_binary" cluster plan --manifest "$manifest" --workers main=4,mini=2 --json)"
  CLUSTER_PLAN_JSON="$json" python3 - <<'PY'
import json
import os
import sys

doc = json.loads(os.environ["CLUSTER_PLAN_JSON"])
runtime = doc.get("runtime_plan")
if doc.get("dry_run") is not True or not isinstance(runtime, dict):
    sys.exit(1)
roles = [stage.get("role") for stage in runtime.get("stages", [])]
if roles != ["embeddings", "transformer_layers", "transformer_layers", "final_norm_head"]:
    sys.exit(1)
if runtime.get("total_layer_count") != 28:
    sys.exit(1)
boundary = doc.get("boundary_tensor")
if not isinstance(boundary, dict):
    sys.exit(1)
if boundary.get("name") != "hidden_states":
    sys.exit(1)
if boundary.get("shape") != [1, -1, 1024]:
    sys.exit(1)
if boundary.get("scalar_type") != "float16":
    sys.exit(1)
if runtime.get("boundary_tensor") != boundary:
    sys.exit(1)
PY
fi

if [[ "$require_ready" == "1" ]]; then
  "$caix_binary" cluster join --help >/dev/null
  "$caix_binary" --help | grep -q -- '--cluster'
fi

echo "brew distributed surface ok"
