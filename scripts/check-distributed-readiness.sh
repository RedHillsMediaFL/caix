#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-distributed-readiness.sh [--caix <path>] [--brew-caix <path>] [--evidence-dir <dir>]
       scripts/check-distributed-readiness.sh --tiny-poc --tiny-manifest <stage-manifest.json> --brew-caix <path> [--caix <path>]

Checks whether distributed inference is ready for Thunderbolt testing.
This is non-heavy: it does not build, load models, start workers, benchmark, download, or upload.
Default mode is the real-Qwen release gate. --tiny-poc is only the tiny staged hardware-POC gate.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
caix_binary="${caix_bin:-}"
brew_caix_binary=""
evidence_dir="$REPO_DIR/docs/distributed-evidence"
expected_prompt_set="docs/distributed-evidence/qwen3-0.6b-prompts.txt"
tiny_poc=0
tiny_manifest=""
not_ready=0
plan_err="$(mktemp "${TMPDIR:-/tmp}/caix-distributed-plan.XXXXXX")"
trap 'rm -f "$plan_err"' EXIT

while [[ $# -gt 0 ]]; do
  case "$1" in
    --caix) caix_binary="${2:?}"; shift 2 ;;
    --brew-caix) brew_caix_binary="${2:?}"; shift 2 ;;
    --evidence-dir) evidence_dir="${2:?}"; shift 2 ;;
    --tiny-poc) tiny_poc=1; shift ;;
    --tiny-manifest) tiny_manifest="${2:?}"; shift 2 ;;
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
if runtime.get("position_mode") != "full_prefix":
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

check_evidence_asset_digests() {
  local label="$1"
  local manifest="$2"
  local digest_file="$3"
  local file="$4"

  if [[ -z "$caix_binary" || ! -x "$caix_binary" ]]; then
    missing "$label evidence asset_digests cannot be validated without caix binary: $file"
    return 1
  fi

  local manifest_path="$REPO_DIR/$manifest"
  local digest_path="$REPO_DIR/$digest_file"
  if "$caix_binary" cluster plan --manifest "$manifest_path" --json >/dev/null 2>"$plan_err"; then
    if REPO_DIR="$REPO_DIR" MANIFEST_PATH="$manifest_path" DIGEST_PATH="$digest_path" python3 - <<'PY'
import hashlib
import json
import os
import pathlib
import sys

repo = pathlib.Path(os.environ["REPO_DIR"]).resolve()
manifest_path = pathlib.Path(os.environ["MANIFEST_PATH"]).resolve()
digest_path = pathlib.Path(os.environ["DIGEST_PATH"]).resolve()


def fail(message):
    print(message, file=sys.stderr)
    sys.exit(1)


def repo_relative(path):
    try:
        return path.resolve().relative_to(repo).as_posix()
    except ValueError:
        fail(f"asset path is outside the repository: {path}")


def asset_digest(path):
    path = path.resolve()
    if not path.exists():
        fail(f"asset path is missing: {path}")
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        return digest.hexdigest()
    if not path.is_dir():
        fail(f"asset path is not a file or directory: {path}")
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        fail(f"asset path has no files to hash: {path}")
    for item in files:
        rel = item.relative_to(path).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def stage_path(stage, keys):
    for key in keys:
        value = stage.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


try:
    manifest = json.loads(manifest_path.read_text())
except Exception as exc:
    fail(f"manifest is not valid JSON: {exc}")
body = manifest.get("cluster") if isinstance(manifest.get("cluster"), dict) else manifest
stages = body.get("stages") if isinstance(body, dict) else None
if not isinstance(stages, list) or not stages:
    fail("manifest has no stages")

expected_assets = []
base = manifest_path.parent
for stage in stages:
    if not isinstance(stage, dict):
        fail("stage entry is not an object")
    main = stage_path(stage, ("bundle", "path", "bundle_path", "aimodel"))
    if main is None:
        fail("stage is missing bundle path")
    expected_assets.append((base / main).resolve())
    decode = stage_path(stage, ("decode_asset", "decode_asset_name", "decode_bundle"))
    if decode is not None:
        expected_assets.append((base / decode).resolve())

declared = {}
for line_no, raw in enumerate(digest_path.read_text().splitlines(), 1):
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    parts = line.split()
    if len(parts) != 2:
        fail(f"{digest_path}: line {line_no} must be '<sha256> <repo-relative-path>'")
    sha, rel = parts
    if len(sha) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in sha):
        fail(f"{digest_path}: line {line_no} has invalid sha256")
    if rel.startswith("/") or "://" in rel or rel in (".", "..") or rel.startswith("../") or "/../" in rel or rel.endswith("/.."):
        fail(f"{digest_path}: line {line_no} path must be repo-relative")
    declared[rel] = sha.lower()

missing = []
mismatched = []
for asset in expected_assets:
    rel = repo_relative(asset)
    actual = asset_digest(asset)
    expected = declared.get(rel)
    if expected is None:
        missing.append(rel)
    elif expected != actual:
        mismatched.append(rel)

if missing:
    fail("asset_digests is missing planned assets: " + ", ".join(missing))
if mismatched:
    fail("asset_digests mismatched planned assets: " + ", ".join(mismatched))
PY
    then
      ready "$label staged asset digests match manifest"
    else
      missing "$label evidence asset_digests do not match planned stage assets: $digest_file"
      return 1
    fi
  else
    missing "$label evidence asset_digests cannot be validated because cluster plan failed: $(tr '\n' ' ' <"$plan_err" | sed 's/[[:space:]]*$//')"
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

  local result mode model manifest caix_commit prompt_set prompts max_tokens temperature token_match asset_digests raw_log
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
  asset_digests="$(evidence_value asset_digests "$file")"
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
  elif [[ "$prompt_set" != "$expected_prompt_set" ]]; then
    missing "$label evidence prompt_set must be $expected_prompt_set: $file"
  elif [[ "$max_tokens" != "128" ]]; then
    missing "$label evidence max_tokens must be 128: $file"
  elif [[ "$temperature" != "0" ]]; then
    missing "$label evidence temperature must be 0: $file"
  elif [[ "$token_match" != "true" ]]; then
    missing "$label evidence token_match must be true: $file"
  elif [[ -z "$asset_digests" ]]; then
    missing "$label evidence asset_digests is missing: $file"
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
  elif ! check_repo_evidence_path "$label" asset_digests "$asset_digests" "$file"; then
    return
  elif ! check_evidence_asset_digests "$label" "$manifest" "$asset_digests" "$file"; then
    return
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

check_tiny_poc_manifest() {
  local manifest="$1"

  if [[ -z "$manifest" ]]; then
    missing "tiny POC manifest is required; pass --tiny-manifest"
    return
  fi
  if [[ ! -f "$manifest" ]]; then
    missing "tiny POC manifest is missing: $manifest"
    return
  fi
  if [[ -z "$caix_binary" || ! -x "$caix_binary" ]]; then
    missing "tiny POC manifest cannot be validated without caix binary"
    return
  fi

  if json="$("$caix_binary" cluster plan \
      --manifest "$manifest" \
      --workers main=4,macbook=2 \
      --json 2>"$plan_err")"; then
    if TINY_PLAN_JSON="$json" python3 - <<'PY'
import json
import os
import sys

doc = json.loads(os.environ["TINY_PLAN_JSON"])
runtime = doc.get("runtime_plan")
if doc.get("dry_run") is not True or not isinstance(runtime, dict):
    sys.exit(1)
if doc.get("model_name") != "qwen3-tiny-random-coreai-staged-rope-input-f16-2x1":
    sys.exit(1)
roles = [stage.get("role") for stage in runtime.get("stages", [])]
if roles != ["embeddings", "transformer_layers", "transformer_layers", "final_norm_head"]:
    sys.exit(1)
if runtime.get("total_layer_count") != 2:
    sys.exit(1)
if runtime.get("position_mode") != "full_prefix":
    sys.exit(1)
boundary = runtime.get("boundary_tensor")
if not isinstance(boundary, dict):
    sys.exit(1)
if boundary.get("name") != "hidden_states":
    sys.exit(1)
if boundary.get("shape") != [1, -1, 64]:
    sys.exit(1)
if boundary.get("scalar_type") != "float16":
    sys.exit(1)
PY
    then
      ready "tiny POC manifest validates with cluster plan"
    else
      missing "tiny POC manifest did not match the expected staged runtime contract"
    fi
  else
    missing "tiny POC manifest failed cluster plan: $(tr '\n' ' ' <"$plan_err" | sed 's/[[:space:]]*$//')"
  fi

  local digest_file
  digest_file="$(mktemp "${TMPDIR:-/tmp}/caix-tiny-poc-digests.XXXXXX")"
  if "$SCRIPT_DIR/check-stage-bundle-copy.sh" --manifest "$manifest" --write "$digest_file" >/dev/null; then
    ready "tiny POC staged assets are digestible for copy verification"
  else
    missing "tiny POC staged assets failed copy-digest verification"
  fi
  rm -f "$digest_file"

  if "$SCRIPT_DIR/check-tiny-cluster-smoke.sh" \
      --caix "$caix_binary" \
      --manifest "$manifest" \
      --coordinator 127.0.0.1:1237 \
      --print-commands >/dev/null; then
    ready "tiny POC cluster smoke commands can be generated"
  else
    missing "tiny POC cluster smoke command generation failed"
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

if [[ "$tiny_poc" == "1" ]]; then
  check_tiny_poc_manifest "$tiny_manifest"
else
  same_machine_evidence="$evidence_dir/same-machine-qwen3-0.6b-token-match.txt"
  loopback_evidence="$evidence_dir/loopback-qwen3-0.6b-token-match.txt"

  check_token_match_evidence "$same_machine_evidence" "same-machine" "same-machine"
  check_token_match_evidence "$loopback_evidence" "loopback" "loopback"
  check_manifest_consistency "$same_machine_evidence" "$loopback_evidence"
fi

if [[ -n "$brew_caix_binary" ]]; then
  brew_manifest="$REPO_DIR/docs/examples/cluster-stage-manifest.json"
  if [[ "$tiny_poc" == "1" ]]; then
    brew_manifest="$tiny_manifest"
  fi
  if "$SCRIPT_DIR/check-brew-distributed.sh" \
      --caix "$brew_caix_binary" \
      --ready \
      --manifest "$brew_manifest" >/dev/null; then
    ready "Brew-installed caix distributed surface passes"
  else
    missing "Brew-installed caix distributed surface failed"
  fi
else
  if [[ "$tiny_poc" == "1" ]]; then
    missing "tiny POC must use Brew-installed caix; pass --brew-caix <path>"
  else
    missing "Brew-installed caix was not checked; pass --brew-caix <path>"
  fi
fi

if [[ "$not_ready" == "0" ]]; then
  if [[ "$tiny_poc" == "1" ]]; then
    echo "tiny distributed POC is ready for Thunderbolt testing"
  else
    echo "distributed is ready for Thunderbolt testing"
  fi
else
  if [[ "$tiny_poc" == "1" ]]; then
    echo "tiny distributed POC is not ready for Thunderbolt testing"
  else
    echo "distributed is not ready for Thunderbolt testing"
  fi
fi

exit "$not_ready"
