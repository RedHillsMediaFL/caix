#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-distributed-readiness.sh [--caix <path>] [--brew-caix <path>] [--evidence-dir <dir>]

Checks whether distributed inference is ready for Thunderbolt testing.
This is non-heavy: it does not build, load models, start workers, benchmark, download, or upload.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
caix_binary="${caix_bin:-}"
brew_caix_binary=""
evidence_dir="$REPO_DIR/docs/distributed-evidence"
not_ready=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --caix) caix_binary="${2:?}"; shift 2 ;;
    --brew-caix) brew_caix_binary="${2:?}"; shift 2 ;;
    --evidence-dir) evidence_dir="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) echo "error: unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$caix_binary" ]]; then
  if [[ -x "$REPO_DIR/.build/debug/caix" ]]; then
    caix_binary="$REPO_DIR/.build/debug/caix"
  elif [[ -x "$REPO_DIR/.build/release/caix" ]]; then
    caix_binary="$REPO_DIR/.build/release/caix"
  elif command -v caix >/dev/null 2>&1; then
    caix_binary="$(command -v caix)"
  fi
fi

ready() {
  printf 'ok: %s\n' "$*"
}

missing() {
  printf 'not ready: %s\n' "$*"
  not_ready=1
}

if [[ -z "$caix_binary" || ! -x "$caix_binary" ]]; then
  missing "caix binary not found; pass --caix"
else
  ready "caix binary: $caix_binary"
  "$caix_binary" --version || missing "caix --version failed"

  if json="$("$caix_binary" cluster plan \
      --manifest "$REPO_DIR/docs/examples/cluster-stage-manifest.json" \
      --workers main=4,mini=2 \
      --json 2>/tmp/caix-distributed-plan.err)"; then
    if CLUSTER_PLAN_JSON="$json" python3 - <<'PY'
import json
import os
import sys

doc = json.loads(os.environ["CLUSTER_PLAN_JSON"])
runtime = doc.get("runtime_plan")
if doc.get("dry_run") is not True or not isinstance(runtime, dict):
    sys.exit(1)
roles = [stage.get("role") for stage in runtime.get("stages", [])]
expected = ["embeddings", "transformer_layers", "transformer_layers", "final_norm_head"]
if roles != expected:
    sys.exit(1)
if runtime.get("total_layer_count") != 28:
    sys.exit(1)
PY
    then
      ready "cluster plan produces validated runtime_plan"
    else
      missing "cluster plan JSON did not match the distributed runtime contract"
    fi
  else
    missing "cluster plan failed: $(tr '\n' ' ' </tmp/caix-distributed-plan.err | sed 's/[[:space:]]*$//')"
  fi

  if "$caix_binary" cluster join --help >/dev/null 2>&1; then
    ready "cluster join CLI exists"
  else
    missing "cluster join CLI is not implemented"
  fi

  if "$caix_binary" --help 2>/dev/null | grep -q -- '--cluster'; then
    ready "serve --cluster is advertised"
  else
    missing "serve --cluster is not implemented"
  fi
fi

same_machine_evidence="$evidence_dir/same-machine-qwen3-0.6b-token-match.txt"
loopback_evidence="$evidence_dir/loopback-qwen3-0.6b-token-match.txt"

if [[ -s "$same_machine_evidence" ]]; then
  ready "same-machine staged Qwen3-0.6B evidence exists"
else
  missing "same-machine staged Qwen3-0.6B token-match evidence is missing: $same_machine_evidence"
fi

if [[ -s "$loopback_evidence" ]]; then
  ready "loopback staged Qwen3-0.6B evidence exists"
else
  missing "loopback staged Qwen3-0.6B token-match evidence is missing: $loopback_evidence"
fi

if [[ -n "$brew_caix_binary" ]]; then
  if "$SCRIPT_DIR/check-brew-distributed.sh" --caix "$brew_caix_binary" >/dev/null; then
    ready "Brew-installed caix distributed surface passes"
  else
    missing "Brew-installed caix distributed surface failed"
  fi
else
  missing "Brew-installed caix was not checked; pass --brew-caix <path>"
fi

if [[ "$not_ready" == "0" ]]; then
  echo "distributed is ready for Thunderbolt testing"
else
  echo "distributed is not ready for Thunderbolt testing"
fi

exit "$not_ready"
