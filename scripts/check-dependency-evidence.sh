#!/usr/bin/env bash
# Verify the committed dependency evidence matches current resolved Core AI dependencies.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-dependency-evidence.sh [--evidence <path>] [--package-resolved <path>] [--coreai-pyproject <path>]

Regenerates benchmarks/DEPENDENCY_EVIDENCE.tsv and fails if it is stale or incomplete.
Does not build, load models, run exports, benchmark, download, or contact external services.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EVIDENCE="$REPO_DIR/benchmarks/DEPENDENCY_EVIDENCE.tsv"
PACKAGE_RESOLVED="$REPO_DIR/Package.resolved"
COREAI_PYPROJECT="$REPO_DIR/.build/checkouts/coreai-models/python/pyproject.toml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --evidence) EVIDENCE="${2:?}"; shift 2 ;;
    --package-resolved) PACKAGE_RESOLVED="${2:?}"; shift 2 ;;
    --coreai-pyproject) COREAI_PYPROJECT="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$EVIDENCE" ]] || { echo "error: dependency evidence file not found: $EVIDENCE" >&2; exit 2; }

tmp="$(mktemp "${TMPDIR:-/tmp}/caix-dependency-evidence-check.XXXXXX")"
trap 'rm -f "$tmp"' EXIT

"$SCRIPT_DIR/collect-dependency-evidence.sh" \
  --package-resolved "$PACKAGE_RESOLVED" \
  --coreai-pyproject "$COREAI_PYPROJECT" \
  --out "$tmp"

if ! cmp -s "$tmp" "$EVIDENCE"; then
  echo "error: $EVIDENCE is stale; regenerate with:" >&2
  echo "  scripts/collect-dependency-evidence.sh --out benchmarks/DEPENDENCY_EVIDENCE.tsv" >&2
  diff -u "$EVIDENCE" "$tmp" >&2 || true
  exit 1
fi

awk -F '\t' '
  NR == 1 {
    expected = "component\tecosystem\tversion_kind\tversion_or_branch\trevision\tlocation\tevidence_source\tnotes"
    if ($0 != expected) {
      printf "error: unexpected header in %s\n", FILENAME > "/dev/stderr"
      bad = 1
    }
    next
  }
  NF != 8 {
    printf "error: %s:%d has %d fields, expected 8\n", FILENAME, NR, NF > "/dev/stderr"
    bad = 1
    next
  }
  $1 == "" || $2 == "" || $3 == "" || $4 == "" || $6 == "" || $7 == "" {
    printf "error: %s:%d has an empty required field\n", FILENAME, NR > "/dev/stderr"
    bad = 1
  }
  $2 == "swiftpm" && $5 !~ /^[0-9a-f]{40}$/ {
    printf "error: %s:%d SwiftPM dependency must record a 40-character revision\n", FILENAME, NR > "/dev/stderr"
    bad = 1
  }
  $2 == "python" && $3 != "exact" {
    printf "error: %s:%d Python Core AI dependencies must use exact versions\n", FILENAME, NR > "/dev/stderr"
    bad = 1
  }
  { seen[$1] = 1 }
  END {
    split("coreai-models xgrammar coreai-torch coreai-core coreai-opt", required, " ")
    for (i in required) {
      if (!seen[required[i]]) {
        printf "error: missing dependency evidence row: %s\n", required[i] > "/dev/stderr"
        bad = 1
      }
    }
    exit bad ? 1 : 0
  }
' "$EVIDENCE"

echo "dependency evidence ok"
