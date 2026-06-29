#!/usr/bin/env bash
# Fail publication checks if local export payloads were left in the checkout.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-export-cleanliness.sh [--exports <dir>]

Checks that models/exports contains only .gitkeep and that no export payloads are tracked.
Does not remove, build, download, upload, or benchmark.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EXPORTS="$REPO_DIR/models/exports"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exports) EXPORTS="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) echo "unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -d "$EXPORTS" ]] || { echo "error: exports directory not found: $EXPORTS" >&2; exit 2; }

EXPORTS_ABS="$(cd "$EXPORTS" && pwd -P)"
REPO_ABS="$(cd "$REPO_DIR" && pwd -P)"
case "$EXPORTS_ABS" in
  "$REPO_ABS"/*) ;;
  *)
    echo "error: exports directory must be inside the repository: $EXPORTS_ABS" >&2
    exit 2
    ;;
esac

exports_rel="${EXPORTS_ABS#"$REPO_ABS"/}"

tracked_payloads="$(
  git -C "$REPO_DIR" ls-files -- "$exports_rel" \
    | awk -v keep="$exports_rel/.gitkeep" '$0 != keep { print }'
)"
if [[ -n "$tracked_payloads" ]]; then
  echo "error: export payload files are tracked; keep models/exports local-only:" >&2
  printf '%s\n' "$tracked_payloads" >&2
  exit 1
fi

if ! git -C "$REPO_DIR" ls-files --error-unmatch -- "$exports_rel/.gitkeep" >/dev/null 2>&1; then
  echo "error: $exports_rel/.gitkeep must be tracked" >&2
  exit 1
fi

leftovers="$(find "$EXPORTS_ABS" -mindepth 1 -maxdepth 1 ! -name .gitkeep -print)"
if [[ -n "$leftovers" ]]; then
  echo "error: export payloads are still present; remove tested bundles with scripts/remove-export.sh:" >&2
  printf '%s\n' "$leftovers" >&2
  exit 1
fi

echo "export cleanliness ok"
