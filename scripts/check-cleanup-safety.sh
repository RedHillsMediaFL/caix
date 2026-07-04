#!/usr/bin/env bash
# Check local export cleanup guards without touching real model payloads.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-cleanup-safety.sh

Exercises scripts/remove-export.sh against a temporary exports directory.
Does not remove real models, build, download, upload, or benchmark.
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
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-cleanup-safety.XXXXXX")"
mkdir -p "$REPO_DIR/.tmp"
report_tmp="$(mktemp -d "$REPO_DIR/.tmp/caix-export-cleanliness.XXXXXX")"
trap 'rm -rf "$tmpdir" "$report_tmp"' EXIT

exports="$tmpdir/exports"
lock="$tmpdir/.agent-heavy-task.lock"
mkdir -p "$exports/demo-bundle" "$exports/keep-bundle"
printf 'demo\n' > "$exports/demo-bundle/marker.txt"
printf 'keep\n' > "$exports/keep-bundle/marker.txt"

env caix_heavy_task_lock="$tmpdir/no-lock" \
  "$SCRIPT_DIR/remove-export.sh" --exports "$exports" --dry-run demo-bundle >/dev/null
[[ -d "$exports/demo-bundle" ]] || {
  echo "error: dry-run removed demo-bundle" >&2
  exit 1
}

touch "$lock"
if env caix_heavy_task_lock="$lock" \
    "$SCRIPT_DIR/remove-export.sh" --exports "$exports" demo-bundle >/dev/null 2>&1; then
  echo "error: remove-export ignored heavy-task lock" >&2
  exit 1
fi
[[ -d "$exports/demo-bundle" ]] || {
  echo "error: locked remove deleted demo-bundle" >&2
  exit 1
}
rm -f "$lock"

if env caix_heavy_task_lock="$lock" \
    "$SCRIPT_DIR/remove-export.sh" --exports "$exports" ../demo-bundle >/dev/null 2>&1; then
  echo "error: remove-export accepted path traversal" >&2
  exit 1
fi
[[ -d "$exports/demo-bundle" ]] || {
  echo "error: traversal probe deleted demo-bundle" >&2
  exit 1
}

env caix_heavy_task_lock="$lock" \
  "$SCRIPT_DIR/remove-export.sh" --exports "$exports" demo-bundle >/dev/null
[[ ! -e "$exports/demo-bundle" ]] || {
  echo "error: unlocked remove left demo-bundle behind" >&2
  exit 1
}
[[ -d "$exports/keep-bundle" ]] || {
  echo "error: unlocked remove touched keep-bundle" >&2
  exit 1
}

report_exports="$report_tmp/exports"
mkdir -p "$report_exports/demo-report"
printf 'demo\n' > "$report_exports/demo-report/marker.txt"

report_output="$("$SCRIPT_DIR/check-export-cleanliness.sh" --exports "$report_exports" --report)"
[[ -d "$report_exports/demo-report" ]] || {
  echo "error: export-cleanliness report removed demo-report" >&2
  exit 1
}
case "$report_output" in
  *"scripts/remove-export.sh --exports"*" --dry-run demo-report"*) ;;
  *)
    echo "error: export-cleanliness report omitted dry-run cleanup command" >&2
    printf '%s\n' "$report_output" >&2
    exit 1
    ;;
esac
case "$report_output" in
  *"report only; no files removed"*) ;;
  *)
    echo "error: export-cleanliness report omitted non-destructive footer" >&2
    printf '%s\n' "$report_output" >&2
    exit 1
    ;;
esac

echo "cleanup safety ok"
