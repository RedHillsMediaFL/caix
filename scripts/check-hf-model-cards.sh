#!/usr/bin/env bash
# Fetch live Hugging Face model cards for manifest repos and run the public-copy guard.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-hf-model-cards.sh [options]

Options:
  --manifest <path>  TSV manifest. Default: benchmarks/MANIFEST.tsv.
  --revision <ref>   Hub ref to fetch. Default: main.

Reads README.md files only through the Hugging Face CLI. Does not download model payloads.
Fails when a live card has public-copy wording that should not ship or omits the support link.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MANIFEST="$REPO_DIR/benchmarks/MANIFEST.tsv"
REVISION="main"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="${2:?}"; shift 2 ;;
    --revision) REVISION="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$MANIFEST" ]] || { echo "error: manifest not found: $MANIFEST" >&2; exit 2; }
command -v hf >/dev/null 2>&1 || { echo "error: hf CLI is required" >&2; exit 2; }

export HF_HOME="${HF_HOME:-/Volumes/SSD/hf-cache}"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-hf-model-cards.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

manifest_repos="$tmpdir/manifest-repos.txt"
cards_dir="$tmpdir/cards"
mkdir -p "$cards_dir"

awk -F '\t' '$1 != "" && $1 != "repo" && $1 !~ /^#/ { print $1 }' "$MANIFEST" \
  | sort -u > "$manifest_repos"

if [[ ! -s "$manifest_repos" ]]; then
  echo "error: manifest has no repos" >&2
  exit 1
fi

missing=0
while IFS= read -r repo; do
  [[ -z "$repo" ]] && continue
  out="$cards_dir/${repo//\//__}.README.md"
  repo_dir="$tmpdir/download/${repo//\//__}"
  if ! hf download "$repo" README.md --revision "$REVISION" --local-dir "$repo_dir" --quiet >/dev/null; then
    echo "error: missing or unreadable model card: $repo@$REVISION" >&2
    missing=1
  elif [[ ! -f "$repo_dir/README.md" ]]; then
    echo "error: README.md missing after download: $repo@$REVISION" >&2
    missing=1
  else
    cp "$repo_dir/README.md" "$out"
  fi
done < "$manifest_repos"

if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

support_missing=0
while IFS= read -r repo; do
  [[ -z "$repo" ]] && continue
  card="$cards_dir/${repo//\//__}.README.md"
  if ! rg -q 'https://redhillsmediafl[.]com/open-source|redhillsmediafl[.]com/open-source' "$card"; then
    echo "error: support link missing from model card: $repo@$REVISION" >&2
    support_missing=1
  fi
done < "$manifest_repos"

if [[ "$support_missing" -ne 0 ]]; then
  exit 1
fi

set +e
guard_output="$("$SCRIPT_DIR/check-public-copy.sh" "$cards_dir" 2>&1)"
guard_status=$?
set -e
if [[ "$guard_status" -ne 0 ]]; then
  printf '%s\n' "$guard_output" \
    | sed "s#$cards_dir/##g" \
    | sed 's#^\([^:]*\)__\([^:]*\)\.README\.md:#\1/\2 README.md:#'
  exit "$guard_status"
fi

count="$(wc -l < "$manifest_repos" | tr -d ' ')"
echo "hf model cards ok: $count cards checked at $REVISION"
