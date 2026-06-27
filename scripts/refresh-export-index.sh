#!/usr/bin/env bash
set -euo pipefail

exports_dir="${1:-${CAIX_EXPORTS:-$PWD/exports}}"
index_path="${2:-${CAIX_EXPORT_INDEX:-$HOME/coreai-server/export-index.json}}"

python3 - "$exports_dir" "$index_path" <<'PY'
import json
import os
import sys
import time

exports_dir = os.path.abspath(sys.argv[1])
index_path = os.path.abspath(os.path.expanduser(sys.argv[2]))

def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None

def is_llm_meta(path):
    meta = read_json(path)
    return isinstance(meta, dict) and meta.get("kind") == "llm"

def mode_for(root):
    if (
        os.path.isdir(os.path.join(root, "eagle_target.aimodel"))
        and os.path.isdir(os.path.join(root, "eagle_draft.aimodel"))
        and os.path.isdir(os.path.join(root, "tokenizer"))
    ):
        return "eagle"
    if not is_llm_meta(os.path.join(root, "metadata.json")):
        return None
    if is_llm_meta(os.path.join(root, "draft", "metadata.json")):
        return "speculative"
    return "standard"

bundles = []
if os.path.isdir(exports_dir):
    for name in sorted(os.listdir(exports_dir)):
        if name.startswith("."):
            continue
        root = os.path.join(exports_dir, name)
        if not os.path.isdir(root):
            continue
        mode = mode_for(root)
        if mode:
            bundles.append({"name": name, "mode": mode})

payload = {
    "version": 1,
    "generatedAt": int(time.time()),
    "exportsDir": exports_dir,
    "bundles": bundles,
}

os.makedirs(os.path.dirname(index_path), exist_ok=True)
tmp_path = index_path + ".tmp"
with open(tmp_path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(tmp_path, index_path)
print(f"indexed {len(bundles)} bundles from {exports_dir} -> {index_path}")
PY
