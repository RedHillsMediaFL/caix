#!/usr/bin/env bash
# Self-test export-cleanliness checks with local fixtures and a temporary Git index.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

tmpdir="$(mktemp -d "$REPO_DIR/.tmp/caix-export-cleanliness-contract.XXXXXX")"
index="$tmpdir/index"
cleanup() {
  rm -rf "$tmpdir"
}
trap cleanup EXIT

clean_exports="$tmpdir/clean-exports"
dirty_exports="$tmpdir/dirty-exports"
tracked_exports="$tmpdir/tracked-exports"
missing_gitkeep_exports="$tmpdir/missing-gitkeep-exports"
mkdir -p "$clean_exports" "$dirty_exports/demo-bundle" "$tracked_exports" "$missing_gitkeep_exports"
printf 'keep\n' > "$clean_exports/.gitkeep"
printf 'keep\n' > "$dirty_exports/.gitkeep"
printf 'payload\n' > "$dirty_exports/demo-bundle/marker.txt"
printf 'keep\n' > "$tracked_exports/.gitkeep"
printf 'tracked payload\n' > "$tracked_exports/payload.txt"

fixture_rel() {
  local path="$1"
  local abs
  abs="$(cd "$path" && pwd -P)"
  printf '%s\n' "${abs#"$REPO_DIR"/}"
}

GIT_INDEX_FILE="$index" git -C "$REPO_DIR" read-tree HEAD
GIT_INDEX_FILE="$index" git -C "$REPO_DIR" add -f \
  "$(fixture_rel "$clean_exports")/.gitkeep" \
  "$(fixture_rel "$dirty_exports")/.gitkeep" \
  "$(fixture_rel "$tracked_exports")/.gitkeep" \
  "$(fixture_rel "$tracked_exports")/payload.txt"

GIT_INDEX_FILE="$index" "$SCRIPT_DIR/check-export-cleanliness.sh" \
  --exports "$clean_exports" >/dev/null

if GIT_INDEX_FILE="$index" "$SCRIPT_DIR/check-export-cleanliness.sh" \
    --exports "$dirty_exports" >"$tmpdir/dirty.out" 2>&1
then
  echo "error: export-cleanliness unexpectedly passed with local payloads present" >&2
  exit 1
fi
grep -F 'export payloads are still present' "$tmpdir/dirty.out" >/dev/null \
  || { echo "error: local-payload failure did not mention present payloads" >&2; cat "$tmpdir/dirty.out" >&2; exit 1; }
grep -F 'check-export-cleanliness.sh --report' "$tmpdir/dirty.out" >/dev/null \
  || { echo "error: local-payload failure did not include report hint" >&2; cat "$tmpdir/dirty.out" >&2; exit 1; }

GIT_INDEX_FILE="$index" "$SCRIPT_DIR/check-export-cleanliness.sh" \
  --exports "$dirty_exports" \
  --report >"$tmpdir/report.out"
[[ -d "$dirty_exports/demo-bundle" ]] \
  || { echo "error: export-cleanliness report removed a payload" >&2; exit 1; }
grep -F 'dry-run cleanup commands:' "$tmpdir/report.out" >/dev/null \
  || { echo "error: report omitted dry-run cleanup commands" >&2; cat "$tmpdir/report.out" >&2; exit 1; }
grep -F 'cleanup commands after dry-run review:' "$tmpdir/report.out" >/dev/null \
  || { echo "error: report omitted post-review cleanup commands" >&2; cat "$tmpdir/report.out" >&2; exit 1; }
grep -F 'report only; no files removed' "$tmpdir/report.out" >/dev/null \
  || { echo "error: report omitted non-destructive footer" >&2; cat "$tmpdir/report.out" >&2; exit 1; }

if GIT_INDEX_FILE="$index" "$SCRIPT_DIR/check-export-cleanliness.sh" \
    --exports "$tracked_exports" >"$tmpdir/tracked.out" 2>&1
then
  echo "error: export-cleanliness unexpectedly passed with tracked payloads" >&2
  exit 1
fi
grep -F 'export payload files are tracked' "$tmpdir/tracked.out" >/dev/null \
  || { echo "error: tracked-payload failure did not mention tracked payloads" >&2; cat "$tmpdir/tracked.out" >&2; exit 1; }

if GIT_INDEX_FILE="$index" "$SCRIPT_DIR/check-export-cleanliness.sh" \
    --exports "$missing_gitkeep_exports" >"$tmpdir/missing-gitkeep.out" 2>&1
then
  echo "error: export-cleanliness unexpectedly passed without tracked .gitkeep" >&2
  exit 1
fi
grep -F '.gitkeep must be tracked' "$tmpdir/missing-gitkeep.out" >/dev/null \
  || { echo "error: missing-gitkeep failure did not mention .gitkeep" >&2; cat "$tmpdir/missing-gitkeep.out" >&2; exit 1; }

echo "export cleanliness contract ok"
