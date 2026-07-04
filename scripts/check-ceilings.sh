#!/usr/bin/env bash
# Verify the no-load decode ceiling artifact is in sync with its assumptions.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-ceilings.sh

Regenerates benchmarks/CEILINGS.md from benchmarks/CEILING_ASSUMPTIONS.tsv and benchmarks/MANIFEST.tsv.
Does not load models, download payloads, benchmark, or contact external services.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
elif [[ "$#" -ne 0 ]]; then
  echo "unknown option: $1" >&2
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

tmp="$(mktemp "${TMPDIR:-/tmp}/caix-ceilings.XXXXXX")"
bundle_tmp="$(mktemp -d "${TMPDIR:-/tmp}/caix-ceiling-bundle.XXXXXX")"
trap 'rm -f "$tmp"; rm -rf "$bundle_tmp"' EXIT

python3 "$SCRIPT_DIR/perf/ceiling.py" \
  --assumptions "$REPO_DIR/benchmarks/CEILING_ASSUMPTIONS.tsv" \
  --manifest "$REPO_DIR/benchmarks/MANIFEST.tsv" \
  --require-estimate qwen3-4b-coreai \
  --require-estimate gemma-4-26b-a4b-coreai \
  --out "$tmp"

if ! cmp -s "$tmp" "$REPO_DIR/benchmarks/CEILINGS.md"; then
  echo "error: benchmarks/CEILINGS.md is stale; regenerate with:" >&2
  echo "  scripts/perf/ceiling.py --out benchmarks/CEILINGS.md" >&2
  exit 1
fi

mkdir -p "$bundle_tmp/exports/demo-coreai/demo.aimodel"
cat > "$bundle_tmp/assumptions.tsv" <<'TSV'
repo	local_dir	kind	benchmark_mode	active_weight_gib	evidence_status	source	notes
example/demo-caix	demo-coreai	standalone	decode	1.00	measured	synthetic	no-load bundle measurement fixture
TSV
cat > "$bundle_tmp/manifest.tsv" <<'TSV'
repo	local_dir	kind	benchmark_mode	status	notes
example/demo-caix	demo-coreai	standalone	decode	eligible	synthetic
TSV
cat > "$bundle_tmp/exports/demo-coreai/metadata.json" <<'JSON'
{"assets":{"main":"demo.aimodel"}}
JSON
dd if=/dev/zero of="$bundle_tmp/exports/demo-coreai/demo.aimodel/main.mlirb" bs=1024 count=1024 >/dev/null 2>&1

python3 "$SCRIPT_DIR/perf/ceiling.py" \
  --assumptions "$bundle_tmp/assumptions.tsv" \
  --manifest "$bundle_tmp/manifest.tsv" \
  --bundle-root "$bundle_tmp/exports" \
  --require-estimate demo-coreai \
  --format tsv \
  --out "$bundle_tmp/report.tsv"

if ! grep -F $'\tdemo.aimodel\tmeasured' "$bundle_tmp/report.tsv" >/dev/null; then
  echo "error: bundle measurement path did not report the synthetic .aimodel asset" >&2
  cat "$bundle_tmp/report.tsv" >&2
  exit 1
fi

echo "ceilings ok"
