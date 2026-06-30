#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-stage-bundle-copy.sh --manifest <stage-manifest.json> (--write <file>|--check <file>) [options]

Writes or verifies SHA-256 digests for a copied staged bundle before a cluster smoke.

Options:
  --manifest <path>       staged manifest
  --remote-stage <id>     transformer stage id to include; repeatable
  --write <file>          write digest list
  --check <file>          verify digest list
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

manifest=""
write_file=""
check_file=""
remote_stage_filters=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) manifest="${2:?}"; shift 2 ;;
    --remote-stage) remote_stage_filters+=("${2:?}"); shift 2 ;;
    --write) write_file="${2:?}"; shift 2 ;;
    --check) check_file="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) echo "error: unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$manifest" ]] || { usage >&2; exit 2; }
if [[ -n "$write_file" && -n "$check_file" ]]; then
  die "choose --write or --check, not both"
fi
[[ -n "$write_file" || -n "$check_file" ]] || {
  echo "error: choose --write or --check" >&2
  usage >&2
  exit 2
}

manifest="$(python3 - "$manifest" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
[[ -f "$manifest" ]] || die "manifest not found: $manifest"

mode="write"
digest_file="$write_file"
if [[ -n "$check_file" ]]; then
  mode="check"
  digest_file="$check_file"
fi

if [[ "$mode" == "write" ]]; then
  mkdir -p "$(dirname "$digest_file")"
else
  [[ -f "$digest_file" ]] || die "digest file not found: $digest_file"
fi

python_args=("$mode" "$manifest" "$digest_file")
if [[ "${#remote_stage_filters[@]}" -gt 0 ]]; then
  python_args+=("${remote_stage_filters[@]}")
fi

python3 - "${python_args[@]}" <<'PY'
from pathlib import Path
import hashlib
import json
import re
import sys


def fail(message):
    raise SystemExit(message)


mode = sys.argv[1]
manifest = Path(sys.argv[2]).expanduser().resolve()
digest_path = Path(sys.argv[3]).expanduser()
filters = set(sys.argv[4:])
base = manifest.parent

with manifest.open("r", encoding="utf-8") as handle:
    doc = json.load(handle)

stages = doc.get("stages")
if stages is None:
    stages = doc.get("cluster", {}).get("stages", [])
if not isinstance(stages, list):
    fail("manifest stages must be an array")


def resolve_under_base(raw, label):
    if not isinstance(raw, str) or not raw:
        fail(f"{label} is missing")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    try:
        rel = path.relative_to(base)
    except ValueError:
        fail(f"{label} is outside manifest root: {path}")
    return path, rel


def add_file(files, path):
    path = path.resolve()
    if not path.is_file():
        fail(f"expected file not found: {path}")
    try:
        rel = path.relative_to(base)
    except ValueError:
        fail(f"file is outside manifest root: {path}")
    files[rel.as_posix()] = path


def add_tree(files, path, label):
    path = path.resolve()
    if path.is_file():
        add_file(files, path)
        return
    if not path.is_dir():
        fail(f"{label} not found: {path}")
    for item in sorted(path.rglob("*")):
        if item.is_file():
            add_file(files, item)


files = {}
add_file(files, manifest)

metadata = base / "metadata.json"
if metadata.exists():
    add_file(files, metadata)

tokenizer = base / "tokenizer"
if tokenizer.exists():
    add_tree(files, tokenizer, "tokenizer")

seen = set()
for stage in stages:
    if not isinstance(stage, dict):
        continue
    stage_id = stage.get("id")
    role = stage.get("role")
    if role != "transformer_layers":
        continue
    if not isinstance(stage_id, str) or not stage_id:
        fail("transformer stage is missing id")
    if filters and stage_id not in filters:
        continue
    seen.add(stage_id)
    stage_path, _ = resolve_under_base(stage.get("bundle"), f"stage {stage_id} bundle")
    add_tree(files, stage_path, f"stage {stage_id} bundle")
    decode_asset = stage.get("decode_asset")
    if decode_asset:
        decode_path, _ = resolve_under_base(decode_asset, f"stage {stage_id} decode_asset")
        add_tree(files, decode_path, f"stage {stage_id} decode_asset")

missing_filters = sorted(filters - seen)
if missing_filters:
    fail("unknown remote stage: " + ", ".join(missing_filters))
if not any(path.startswith("stages/") for path in files):
    fail("manifest has no selected transformer stage assets")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


expected = {rel: digest(path) for rel, path in sorted(files.items())}

if mode == "write":
    with digest_path.open("w", encoding="utf-8") as handle:
        for rel, hex_digest in sorted(expected.items()):
            handle.write(f"{hex_digest}  {rel}\n")
    print(f"stage bundle copy digests wrote: {len(expected)} files {digest_path}")
    raise SystemExit(0)

if mode != "check":
    fail(f"unknown mode: {mode}")

actual = {}
line_re = re.compile(r"^([0-9a-f]{64})\s+(.+)$")
with digest_path.open("r", encoding="utf-8") as handle:
    for line_no, raw in enumerate(handle, 1):
        line = raw.rstrip("\n")
        if not line:
            continue
        match = line_re.match(line)
        if not match:
            fail(f"{digest_path}: line {line_no} must be '<sha256> <relative-path>'")
        rel = match.group(2)
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            fail(f"{digest_path}: line {line_no} path must be relative and stay under manifest root")
        actual[rel] = match.group(1)

missing = sorted(set(expected) - set(actual))
extra = sorted(set(actual) - set(expected))
mismatched = sorted(rel for rel in set(expected) & set(actual) if expected[rel] != actual[rel])

if missing:
    fail("digest file is missing copied assets: " + ", ".join(missing[:10]))
if extra:
    fail("digest file has assets outside the selected manifest set: " + ", ".join(extra[:10]))
if mismatched:
    fail("copied asset digest mismatch: " + ", ".join(mismatched[:10]))

print(f"stage bundle copy ok: {len(expected)} files")
PY
