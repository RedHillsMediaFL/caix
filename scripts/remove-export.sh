#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/remove-export.sh [--exports <dir>] [--dry-run] <bundle-name>

Removes one local bundle under models/exports. Refuses to run while .agent-heavy-task.lock exists.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPORTS="$REPO_DIR/models/exports"
LOCK="${CAIX_HEAVY_TASK_LOCK:-$REPO_DIR/.agent-heavy-task.lock}"
DRY_RUN=0
NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exports) EXPORTS="${2:?}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [[ -n "$NAME" ]]; then
        echo "error: only one bundle name is accepted" >&2
        exit 2
      fi
      NAME="$1"
      shift
      ;;
  esac
done

[[ -n "$NAME" ]] || { echo "error: bundle name is required" >&2; usage >&2; exit 2; }
[[ "$NAME" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "error: bundle name must contain only letters, digits, dot, underscore, or dash" >&2
  exit 2
}
[[ -d "$EXPORTS" ]] || { echo "error: exports directory not found: $EXPORTS" >&2; exit 2; }
if [[ -e "$LOCK" ]]; then
  echo "error: refusing to remove exports while heavy-task lock exists: $LOCK" >&2
  exit 2
fi

EXPORTS_ABS="$(cd "$EXPORTS" && pwd -P)"
TARGET="$EXPORTS_ABS/$NAME"
[[ -d "$TARGET" ]] || { echo "error: bundle not found: $TARGET" >&2; exit 2; }

if [[ "$DRY_RUN" == "1" ]]; then
  echo "would remove $TARGET"
else
  rm -rf -- "$TARGET"
fi
