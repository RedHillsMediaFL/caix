#!/usr/bin/env bash
# Check conversion-guard lock handling without touching real model payloads.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-conversion-guard.sh

Exercises scripts/conversion-guard.sh against a temporary heavy-task lock.
Does not build, convert, download, upload, benchmark, or touch models/exports.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
elif [[ "$#" -ne 0 ]]; then
  echo "unknown option: $1" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-conversion-guard.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

lock="$tmpdir/.agent-heavy-task.lock"
touch "$lock"

if env caix_heavy_task_lock="$lock" \
    "$SCRIPT_DIR/conversion-guard.sh" >"$tmpdir/out.txt" 2>&1; then
  echo "error: conversion-guard ignored heavy-task lock" >&2
  exit 1
fi
if ! grep -F "heavy-task lock exists: $lock" "$tmpdir/out.txt" >/dev/null; then
  echo "error: conversion-guard did not report heavy-task lock" >&2
  exit 1
fi

if env caix_heavy_task_lock="$lock" \
    "$SCRIPT_DIR/conversion-guard.sh" --json >"$tmpdir/status.json"; then
  echo "error: conversion-guard JSON mode ignored heavy-task lock" >&2
  exit 1
fi
json="$(<"$tmpdir/status.json")"
JSON="$json" LOCK="$lock" python3 - <<'PY'
import json
import os
import sys

doc = json.loads(os.environ["JSON"])
if doc.get("busy") is not True:
    sys.exit("busy must be true")
if doc.get("lock_path") != os.environ["LOCK"]:
    sys.exit("lock_path mismatch")
if not isinstance(doc.get("jobs"), list):
    sys.exit("jobs must be a list")
PY

if ! env caix_heavy_task_lock="$lock" \
    "$SCRIPT_DIR/conversion-guard.sh" --ignore-lock >"$tmpdir/ignore-lock.txt" 2>&1; then
  echo "error: conversion-guard --ignore-lock rejected a lock-only state" >&2
  exit 1
fi
if ! grep -F "idle" "$tmpdir/ignore-lock.txt" >/dev/null; then
  echo "error: conversion-guard --ignore-lock did not report idle" >&2
  exit 1
fi

echo "conversion guard ok"
