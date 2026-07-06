#!/usr/bin/env bash
# Validate the local repo<TAB>revision table used for benchmark/tester requests.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-model-revisions.sh [options]

Options:
  --manifest <path>   TSV manifest. Default: benchmarks/MANIFEST.tsv.
  --revisions <path>  repo<TAB>revision TSV. Default: benchmarks/revisions.tsv.

Does not touch the Hub. Fails when revisions are malformed, missing manifest repos,
duplicated, or stale relative to the benchmark manifest.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MANIFEST="$REPO_DIR/benchmarks/MANIFEST.tsv"
REVISIONS="$REPO_DIR/benchmarks/revisions.tsv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="${2:?}"; shift 2 ;;
    --revisions) REVISIONS="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$MANIFEST" ]] || { echo "error: manifest not found: $MANIFEST" >&2; exit 2; }
if [[ ! -f "$REVISIONS" ]]; then
  details="${REVISIONS%.tsv}-details.tsv"
  echo "error: revisions file not found: $REVISIONS" >&2
  echo "hint: run scripts/collect-model-revisions.sh --manifest \"$MANIFEST\" --out \"$REVISIONS\" --details \"$details\"" >&2
  exit 2
fi

awk -F '\t' -v revisions_file="$REVISIONS" -v manifest_file="$MANIFEST" '
  function fail(message) {
    print "error: " message > "/dev/stderr"
    failed = 1
  }
  FILENAME == revisions_file {
    if ($0 == "" || $0 ~ /^#/) next
    revision_rows++
    if (NF != 2) {
      fail(FILENAME ":" FNR " must be repo<TAB>revision")
      next
    }
    if ($1 == "" || $2 == "") {
      fail(FILENAME ":" FNR " has an empty repo or revision")
      next
    }
    if ($2 !~ /^[0-9a-f]{40}$/) {
      fail(FILENAME ":" FNR " revision for " $1 " is not a 40-character lowercase SHA")
    }
    if ($1 in revisions) {
      fail(FILENAME ":" FNR " duplicates repo " $1)
    }
    revisions[$1] = $2
    next
  }
  FILENAME == manifest_file {
    if ($1 == "" || $1 == "repo" || $1 ~ /^#/) next
    manifest_rows++
    manifest[$1] = 1
    if (!($1 in revisions)) {
      fail("missing revision for manifest repo " $1)
    }
    next
  }
  END {
    if (revision_rows == 0) {
      fail("no revisions found in " revisions_file)
    }
    if (manifest_rows == 0) {
      fail("no manifest repos found in " manifest_file)
    }
    for (repo in revisions) {
      if (!(repo in manifest)) {
        fail("stale revision for repo not in manifest: " repo)
      }
    }
    if (failed) exit 1
    printf "model revisions ok: %d manifest repos, %d revisions\n", manifest_rows, revision_rows
  }
' "$REVISIONS" "$MANIFEST"
