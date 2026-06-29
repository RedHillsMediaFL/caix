#!/usr/bin/env bash
# Check that docs/TESTER_REQUESTS.md matches the generated tester request sheet.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-tester-requests.sh [options]

Options:
  --manifest <path>   TSV manifest. Default: benchmarks/MANIFEST.tsv.
  --revisions <path>  Optional repo<TAB>revision TSV. Default: generator default.
  --raw-dir <path>    Raw benchmark root. Default: benchmarks/raw.
  --doc <path>        Tester request markdown. Default: docs/TESTER_REQUESTS.md.

Does not download models. Fails when the committed tester request sheet is stale.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MANIFEST="$REPO_DIR/benchmarks/MANIFEST.tsv"
REVISIONS=""
RAW_DIR="$REPO_DIR/benchmarks/raw"
DOC="$REPO_DIR/docs/TESTER_REQUESTS.md"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="${2:?}"; shift 2 ;;
    --revisions) REVISIONS="${2:?}"; shift 2 ;;
    --raw-dir) RAW_DIR="${2:?}"; shift 2 ;;
    --doc) DOC="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$DOC" ]] || { echo "error: tester request doc not found: $DOC" >&2; exit 2; }

tmp="$(mktemp "${TMPDIR:-/tmp}/caix-tester-requests.XXXXXX.md")"
trap 'rm -f "$tmp"' EXIT

args=(--manifest "$MANIFEST" --raw-dir "$RAW_DIR" --out "$tmp")
if [[ -n "$REVISIONS" ]]; then
  args+=(--revisions "$REVISIONS")
fi

"$SCRIPT_DIR/generate-tester-requests.sh" "${args[@]}" >/dev/null

if ! diff -u "$DOC" "$tmp"; then
  echo "error: tester request sheet is stale; run scripts/generate-tester-requests.sh --out $DOC" >&2
  exit 1
fi

echo "tester requests ok"
