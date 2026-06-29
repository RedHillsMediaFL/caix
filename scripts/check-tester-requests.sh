#!/usr/bin/env bash
# Check that docs/TESTER_REQUESTS.md matches the generated tester request sheet.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-tester-requests.sh [options]

Options:
  --manifest <path>   TSV manifest. Default: benchmarks/MANIFEST.tsv.
  --revisions <path>  Optional repo<TAB>revision TSV. Default: source declared in doc.
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

if [[ -z "$REVISIONS" ]]; then
  doc_revision_source="$(sed -n 's/^Revision source: `\([^`]*\)`\.$/\1/p' "$DOC" | head -n 1)"
  if [[ -n "$doc_revision_source" && "$doc_revision_source" != "none" ]]; then
    if [[ "$doc_revision_source" = /* ]]; then
      inferred_revisions="$doc_revision_source"
    else
      inferred_revisions="$REPO_DIR/$doc_revision_source"
    fi
    if [[ ! -f "$inferred_revisions" ]]; then
      echo "error: tester request doc declares missing revision source: $doc_revision_source" >&2
      echo "run scripts/collect-model-revisions.sh --out $doc_revision_source or pass --revisions <path>" >&2
      exit 1
    fi
    REVISIONS="$inferred_revisions"
  fi
fi

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-tester-requests.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT
tmp="$tmpdir/TESTER_REQUESTS.md"

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
