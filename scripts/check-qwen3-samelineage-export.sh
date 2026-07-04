#!/usr/bin/env bash
# Check same-lineage P0 export wrapper lifecycle without running model exports.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-qwen3-samelineage-export.sh

Exercises scripts/export-qwen3-samelineage-p0.sh in dry-run mode against a
temporary exporter root and heavy lock. Does not build, download, convert,
upload, benchmark, or touch models/exports.
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

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-qwen3-samelineage-export.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

exporter_root="$tmpdir/exporter"
output_dir="$tmpdir/evidence"
lock="$tmpdir/.agent-heavy-task.lock"
mkdir -p "$exporter_root"

reported="$(CAIX_MIN_FREE_RAM_GB=12 CAIX_MIN_FREE_DISK_GB=100 "$SCRIPT_DIR/export-qwen3-samelineage-p0.sh" \
  --dry-run \
  --exporter-root "$exporter_root" \
  --lock "$lock" \
  --output-dir "$output_dir")"

if [[ "$reported" != "$output_dir" ]]; then
  echo "error: export wrapper did not echo output directory" >&2
  echo "expected: $output_dir" >&2
  echo "observed: $reported" >&2
  exit 1
fi
if [[ -e "$lock" ]]; then
  echo "error: export wrapper left heavy lock behind" >&2
  exit 1
fi
if [[ ! -f "$output_dir/export.meta" || ! -f "$output_dir/export.log" ]]; then
  echo "error: export wrapper did not write expected evidence files" >&2
  exit 1
fi

for expected in \
  "step=qwen3-samelineage-noopt-export" \
  "dry_run=1" \
  "min_free_ram_gb=12" \
  "min_free_disk_gb=100" \
  "exit_status=0" \
  "monolithic_output=$exporter_root/exports/qwen3-0.6b-coreai-f16-extnone-noopt-samelineage" \
  "staged_output=$exporter_root/exports/qwen3-0.6b-coreai-staged-f16-extnone-noopt-first3-samelineage" \
  "stage_splits=0:1,1:2,2:3,3:28"; do
  if ! grep -F -- "$expected" "$output_dir/export.meta" >/dev/null; then
    echo "error: export.meta missing: $expected" >&2
    exit 1
  fi
done
if ! grep -E '^ended_utc=[0-9]{8}T[0-9]{6}Z$' "$output_dir/export.meta" >/dev/null; then
  echo "error: export.meta missing ended_utc footer" >&2
  exit 1
fi
for expected in "dry_run=1" "monolithic_exit=0" "staged_exit=0"; do
  if ! grep -F -- "$expected" "$output_dir/export.log" >/dev/null; then
    echo "error: export.log missing: $expected" >&2
    exit 1
  fi
done

if "$SCRIPT_DIR/export-qwen3-samelineage-p0.sh" \
    --exporter-root "$exporter_root" \
    --python /usr/bin/true \
    --lock "$lock" \
    --output-dir "$tmpdir/resource-floor-evidence" \
    --min-free-ram-gb 999999 \
    --min-free-disk-gb 1 >"$tmpdir/resource-floor.out" 2>"$tmpdir/resource-floor.err"; then
  echo "error: export wrapper ignored real-export RAM floor" >&2
  exit 1
fi
if ! grep -F "error: free RAM " "$tmpdir/resource-floor.err" >/dev/null; then
  echo "error: export wrapper did not report real-export RAM floor" >&2
  exit 1
fi
if [[ -e "$lock" ]]; then
  echo "error: export wrapper left heavy lock behind after resource-floor failure" >&2
  exit 1
fi

touch "$lock"
if "$SCRIPT_DIR/export-qwen3-samelineage-p0.sh" \
    --dry-run \
    --exporter-root "$exporter_root" \
    --lock "$lock" \
    --output-dir "$tmpdir/locked-evidence" >"$tmpdir/locked.out" 2>"$tmpdir/locked.err"; then
  echo "error: export wrapper ignored existing heavy lock" >&2
  exit 1
fi
if ! grep -F "error: heavy lock exists: $lock" "$tmpdir/locked.err" >/dev/null; then
  echo "error: export wrapper did not report existing heavy lock" >&2
  exit 1
fi
rm -f "$lock"

mkdir -p "$tmpdir/nonempty"
printf 'occupied\n' > "$tmpdir/nonempty/marker.txt"
if "$SCRIPT_DIR/export-qwen3-samelineage-p0.sh" \
    --dry-run \
    --exporter-root "$exporter_root" \
    --lock "$lock" \
    --output-dir "$tmpdir/nonempty" >"$tmpdir/nonempty.out" 2>"$tmpdir/nonempty.err"; then
  echo "error: export wrapper accepted non-empty evidence directory" >&2
  exit 1
fi
if ! grep -F "error: output directory already exists and is not empty" "$tmpdir/nonempty.err" >/dev/null; then
  echo "error: export wrapper did not report non-empty evidence directory" >&2
  exit 1
fi

echo "qwen3 same-lineage export wrapper ok"
