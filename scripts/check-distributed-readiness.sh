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
plan_err="$(mktemp "${TMPDIR:-/tmp}/caix-distributed-plan.XXXXXX")"
trap 'rm -f "$plan_err"' EXIT

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

evidence_value() {
  local key="$1"
  local file="$2"
  awk -v key="$key" '
    index($0, key "=") == 1 {
      sub("^[^=]*=", "")
      print
      exit
    }
  ' "$file"
}

check_repo_evidence_path() {
  local label="$1"
  local field="$2"
  local value="$3"
  local file="$4"

  if [[ -z "$value" ]]; then
    missing "$label evidence $field is missing: $file"
    return 1
  fi

  if [[ "$value" == /* || "$value" == *://* ||
        "$value" == "." || "$value" == ".." ||
        "$value" == ../* || "$value" == */../* || "$value" == */.. ]]; then
    missing "$label evidence $field must be a repo-relative tracked path: $value"
    return 1
  fi

  if [[ ! -e "$REPO_DIR/$value" ]]; then
    missing "$label evidence $field path is missing: $value"
    return 1
  fi

  if ! git -C "$REPO_DIR" ls-files --error-unmatch -- "$value" >/dev/null 2>&1; then
    missing "$label evidence $field path is not tracked: $value"
    return 1
  fi
}

check_evidence_manifest_plan() {
  local label="$1"
  local manifest="$2"
  local file="$3"

  if [[ -z "$caix_binary" || ! -x "$caix_binary" ]]; then
    missing "$label evidence manifest cannot be validated without caix binary: $file"
    return 1
  fi

  local manifest_path="$REPO_DIR/$manifest"
  if json="$("$caix_binary" cluster plan --manifest "$manifest_path" --json 2>"$plan_err")"; then
    if EVIDENCE_MANIFEST_JSON="$json" python3 - <<'PY'
import json
import os
import sys

doc = json.loads(os.environ["EVIDENCE_MANIFEST_JSON"])
runtime = doc.get("runtime_plan")
if doc.get("dry_run") is not True or not isinstance(runtime, dict):
    sys.exit(1)
roles = [stage.get("role") for stage in runtime.get("stages", [])]
if len(roles) < 3 or roles[0] != "embeddings" or roles[-1] != "final_norm_head":
    sys.exit(1)
if any(role != "transformer_layers" for role in roles[1:-1]):
    sys.exit(1)
if not isinstance(runtime.get("total_layer_count"), int) or runtime["total_layer_count"] <= 0:
    sys.exit(1)
if runtime.get("position_mode") not in ("full_prefix", "current"):
    sys.exit(1)
boundary = runtime.get("boundary_tensor")
if not isinstance(boundary, dict):
    sys.exit(1)
if boundary.get("name") != "hidden_states":
    sys.exit(1)
shape = boundary.get("shape")
if not (
    isinstance(shape, list)
    and len(shape) == 3
    and shape[0] == 1
    and (shape[1] == -1 or (isinstance(shape[1], int) and shape[1] > 0))
    and isinstance(shape[2], int)
    and shape[2] > 0
):
    sys.exit(1)
if boundary.get("scalar_type") not in ("float16", "float32"):
    sys.exit(1)
PY
    then
      ready "$label evidence manifest validates with cluster plan"
    else
      missing "$label evidence manifest did not produce a valid runtime_plan: $manifest"
      return 1
    fi
  else
    missing "$label evidence manifest failed cluster plan: $(tr '\n' ' ' <"$plan_err" | sed 's/[[:space:]]*$//')"
    return 1
  fi
}

prompt_count() {
  local path="$1"
  awk 'NF { count += 1 } END { print count + 0 }' "$path"
}

check_token_match_evidence() {
  local file="$1"
  local label="$2"
  local expected_mode="$3"

  if [[ ! -s "$file" ]]; then
    missing "$label token-match evidence is missing: $file"
    return
  fi

  local result mode model manifest caix_commit prompt_set prompts max_tokens temperature token_match raw_log
  result="$(evidence_value result "$file")"
  mode="$(evidence_value mode "$file")"
  model="$(evidence_value model "$file")"
  manifest="$(evidence_value manifest "$file")"
  caix_commit="$(evidence_value caix_commit "$file")"
  prompt_set="$(evidence_value prompt_set "$file")"
  prompts="$(evidence_value prompts "$file")"
  max_tokens="$(evidence_value max_tokens "$file")"
  temperature="$(evidence_value temperature "$file")"
  token_match="$(evidence_value token_match "$file")"
  raw_log="$(evidence_value raw_log "$file")"

  if [[ "$result" != "pass" ]]; then
    missing "$label evidence result must be pass: $file"
  elif [[ "$mode" != "$expected_mode" ]]; then
    missing "$label evidence mode must be $expected_mode: $file"
  elif [[ "$model" != "qwen3-0.6b-coreai" ]]; then
    missing "$label evidence model must be qwen3-0.6b-coreai: $file"
  elif [[ -z "$manifest" ]]; then
    missing "$label evidence manifest is missing: $file"
  elif [[ ! "$caix_commit" =~ ^[0-9a-f]{40}$ ]]; then
    missing "$label evidence caix_commit must be a 40-character SHA: $file"
  elif ! git -C "$REPO_DIR" cat-file -e "$caix_commit^{commit}" 2>/dev/null; then
    missing "$label evidence caix_commit is not present in this repository: $caix_commit"
  elif [[ ! "$prompts" =~ ^[1-9][0-9]*$ ]]; then
    missing "$label evidence prompts must be a positive integer: $file"
  elif [[ "$max_tokens" != "128" ]]; then
    missing "$label evidence max_tokens must be 128: $file"
  elif [[ "$temperature" != "0" ]]; then
    missing "$label evidence temperature must be 0: $file"
  elif [[ "$token_match" != "true" ]]; then
    missing "$label evidence token_match must be true: $file"
  elif [[ -z "$raw_log" ]]; then
    missing "$label evidence raw_log is missing: $file"
  elif ! check_repo_evidence_path "$label" manifest "$manifest" "$file"; then
    return
  elif ! check_evidence_manifest_plan "$label" "$manifest" "$file"; then
    return
  elif ! check_repo_evidence_path "$label" prompt_set "$prompt_set" "$file"; then
    return
  elif [[ "$(prompt_count "$REPO_DIR/$prompt_set")" != "$prompts" ]]; then
    missing "$label evidence prompts=$prompts does not match prompt_set line count: $prompt_set"
  elif ! check_repo_evidence_path "$label" raw_log "$raw_log" "$file"; then
    return
  else
    ready "$label staged Qwen3-0.6B evidence is structured"
  fi
}

check_manifest_consistency() {
  local same_machine_file="$1"
  local loopback_file="$2"

  if [[ ! -s "$same_machine_file" || ! -s "$loopback_file" ]]; then
    return 0
  fi

  local same_machine_manifest loopback_manifest
  same_machine_manifest="$(evidence_value manifest "$same_machine_file")"
  loopback_manifest="$(evidence_value manifest "$loopback_file")"

  [[ -n "$same_machine_manifest" && -n "$loopback_manifest" ]] || return

  if [[ "$same_machine_manifest" == "$loopback_manifest" ]]; then
    ready "same-machine and loopback evidence use the same manifest"
  else
    missing "same-machine and loopback evidence manifests differ"
  fi
}

if [[ -z "$caix_binary" || ! -x "$caix_binary" ]]; then
  missing "caix binary not found; pass --caix"
else
  ready "caix binary: $caix_binary"
  "$caix_binary" --version || missing "caix --version failed"

  if json="$("$caix_binary" cluster plan \
      --manifest "$REPO_DIR/docs/examples/cluster-stage-manifest.json" \
      --workers main=4,mini=2 \
      --json 2>"$plan_err")"; then
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
if doc.get("position_mode") != "full_prefix":
    sys.exit(1)
if runtime.get("position_mode") != doc["position_mode"]:
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
    then
      ready "cluster plan produces validated runtime_plan and boundary_tensor"
    else
      missing "cluster plan JSON did not match the distributed runtime contract"
    fi
  else
    missing "cluster plan failed: $(tr '\n' ' ' <"$plan_err" | sed 's/[[:space:]]*$//')"
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

check_token_match_evidence "$same_machine_evidence" "same-machine" "same-machine"
check_token_match_evidence "$loopback_evidence" "loopback" "loopback"
check_manifest_consistency "$same_machine_evidence" "$loopback_evidence"

if [[ -n "$brew_caix_binary" ]]; then
  if "$SCRIPT_DIR/check-brew-distributed.sh" \
      --caix "$brew_caix_binary" \
      --ready \
      --manifest "$REPO_DIR/docs/examples/cluster-stage-manifest.json" >/dev/null; then
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
