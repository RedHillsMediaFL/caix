#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-converter-cluster.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

bundle="$tmpdir/qwen3-0.6b-coreai"
mkdir -p "$bundle/tokenizer" "$bundle/stages"
printf '{}\n' > "$bundle/tokenizer/tokenizer.json"
cat > "$bundle/metadata.json" <<'JSON'
{
  "metadata_version": "0.2",
  "kind": "llm",
  "name": "qwen3-0.6b-coreai",
  "assets": {"main": "model.aimodel"},
  "language": {
    "tokenizer": "tokenizer",
    "vocab_size": 151936,
    "max_context_length": 8192,
    "embedded_tokenizer": true
  },
  "source": {"hf_model_id": "Qwen/Qwen3-0.6B"}
}
JSON

for asset in \
  model.aimodel \
  stages/00-embed.aimodel \
  stages/00-embed-decode.aimodel \
  stages/01-layers.aimodel \
  stages/02-head.aimodel; do
  mkdir -p "$bundle/$asset"
  printf 'mlir\n' > "$bundle/$asset/main.mlirb"
done

manifest="$tmpdir/stage-manifest.json"
cat > "$manifest" <<'JSON'
{
  "schema": "caix.cluster.stage_manifest.v0",
  "model": "qwen3-0.6b-coreai",
  "total_layer_count": 28,
  "position_mode": "full_prefix",
  "boundary": {
    "hidden_state": {
      "name": "hidden_states",
      "shape": [1, -1, 1024],
      "scalar_type": "float16"
    }
  },
  "stages": [
    {
      "id": "embed",
      "role": "embeddings",
      "layers": "embeddings",
      "bundle": "stages/00-embed.aimodel",
      "decode_asset": "stages/00-embed-decode.aimodel",
      "function_map": {"main": ["main"], "decode": ["decode"]},
      "vocab_size": 151936,
      "memory_gb": 1.0
    },
    {
      "id": "layers-00-28",
      "role": "transformer_layers",
      "layers": [0, 28],
      "bundle": "stages/01-layers.aimodel",
      "function_map": {"main": ["main"]},
      "memory_gb": 2.0
    },
    {
      "id": "head",
      "role": "final_norm_head",
      "layers": "norm+lm_head",
      "bundle": "stages/02-head.aimodel",
      "function_map": {"main": ["main"]},
      "vocab_size": 151936,
      "memory_gb": 1.0
    }
  ]
}
JSON

python3 "$REPO_DIR/python/converter/convert.py" \
  --bundle "$bundle" \
  --attach-cluster-manifest "$manifest" >/dev/null

python3 - "$bundle/metadata.json" <<'PY'
import json
import sys
meta = json.load(open(sys.argv[1]))
cluster = meta.get("cluster")
if not isinstance(cluster, dict):
    sys.exit("missing cluster block")
if cluster.get("schema") != "caix.cluster.stage_manifest.v0":
    sys.exit("bad schema")
if cluster.get("total_layer_count") != 28:
    sys.exit("bad total layer count")
if cluster.get("position_mode") != "full_prefix":
    sys.exit("bad position mode")
if cluster.get("boundary", {}).get("hidden_state", {}).get("shape") != [1, -1, 1024]:
    sys.exit("bad boundary")
if [stage.get("role") for stage in cluster.get("stages", [])] != [
    "embeddings",
    "transformer_layers",
    "final_norm_head",
]:
    sys.exit("bad stage roles")
PY

rm -rf "$bundle/stages/02-head.aimodel"
if python3 "$REPO_DIR/python/converter/convert.py" \
    --bundle "$bundle" \
    --attach-cluster-manifest "$manifest" >/dev/null 2>&1; then
  echo "error: converter accepted a missing staged asset" >&2
  exit 1
fi

echo "converter cluster metadata ok"
