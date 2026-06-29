#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib/caix-env.sh"

caix_bin="$(caix_env caix_bin BIN "$REPO_DIR/.build/debug/caix")"
MANIFEST="${1:-$REPO_DIR/docs/examples/cluster-stage-manifest.json}"

if [[ ! -x "$caix_bin" ]]; then
  swift build --product caix >/dev/null
fi

json="$("$caix_bin" cluster plan \
  --manifest "$MANIFEST" \
  --workers main=4,mini=2 \
  --json)"

CLUSTER_PLAN_JSON="$json" python3 - <<'PY'
import json
import os
import sys

doc = json.loads(os.environ["CLUSTER_PLAN_JSON"])

def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)

if doc.get("dry_run") is not True:
    fail("cluster plan must be dry_run")
if doc.get("model_name") != "qwen3-0.6b-coreai":
    fail("unexpected model_name")
if doc.get("total_layer_count") != 28:
    fail("unexpected total_layer_count")

runtime = doc.get("runtime_plan")
if not isinstance(runtime, dict):
    fail("missing runtime_plan")
if runtime.get("model_name") != doc["model_name"]:
    fail("runtime_plan model_name drift")
if runtime.get("total_layer_count") != doc["total_layer_count"]:
    fail("runtime_plan total_layer_count drift")
if doc.get("position_mode") != "full_prefix":
    fail("unexpected position_mode")
if runtime.get("position_mode") != doc["position_mode"]:
    fail("runtime_plan position_mode drift")

boundary = doc.get("boundary_tensor")
if not isinstance(boundary, dict):
    fail("missing boundary_tensor")
if boundary.get("name") != "hidden_states":
    fail("unexpected boundary tensor name")
if boundary.get("shape") != [1, -1, 1024]:
    fail("unexpected boundary tensor shape")
if boundary.get("scalar_type") != "float16":
    fail("unexpected boundary tensor scalar_type")
if runtime.get("boundary_tensor") != boundary:
    fail("runtime_plan boundary_tensor drift")

stages = runtime.get("stages")
if not isinstance(stages, list) or len(stages) != 4:
    fail("runtime_plan stages must contain four rows")
roles = [stage.get("role") for stage in stages]
if roles != ["embeddings", "transformer_layers", "transformer_layers", "final_norm_head"]:
    fail(f"unexpected stage roles: {roles}")
if stages[1].get("layer_range") != {"lower_bound": 0, "upper_bound": 14}:
    fail("first layer range mismatch")
if stages[2].get("layer_range") != {"lower_bound": 14, "upper_bound": 28}:
    fail("second layer range mismatch")
if any("Derived" in warning for warning in doc.get("warnings", [])):
    fail("example manifest should use explicit total_layer_count")

print("cluster plan ok")
PY
