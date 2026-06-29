#!/usr/bin/env bash
# Check that live RHM caix family collections cover the manifest and use public-safe notes.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-hf-collections.sh [options]

Options:
  --manifest <path>  TSV manifest. Default: benchmarks/MANIFEST.tsv.
  --owner <name>     Hugging Face owner. Default: redhillsmediafl.
  --limit <n>        Max collections to inspect. Default: 50.

Reads Hugging Face metadata only. Does not download model files.
Fails when a manifest repo is missing from live family collections or a collection note contains
public-copy wording that should not ship.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MANIFEST="$REPO_DIR/benchmarks/MANIFEST.tsv"
OWNER="redhillsmediafl"
LIMIT=50

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="${2:?}"; shift 2 ;;
    --owner) OWNER="${2:?}"; shift 2 ;;
    --limit) LIMIT="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$MANIFEST" ]] || { echo "error: manifest not found: $MANIFEST" >&2; exit 2; }
[[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "error: --limit must be a positive integer" >&2; exit 2; }
command -v curl >/dev/null 2>&1 || { echo "error: curl is required" >&2; exit 2; }
command -v hf >/dev/null 2>&1 || { echo "error: hf CLI not found" >&2; exit 2; }
command -v jq >/dev/null 2>&1 || { echo "error: jq is required to parse Hub JSON" >&2; exit 2; }

export HF_HOME="${HF_HOME:-/Volumes/SSD/hf-cache}"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-hf-collections.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

collections_json="$tmpdir/collections.json"
slugs="$tmpdir/slugs.txt"
items="$tmpdir/items.tsv"
manifest_repos="$tmpdir/manifest-repos.txt"
collection_repos="$tmpdir/collection-repos.txt"

HF_HUB_DISABLE_PROGRESS_BARS=1 hf collections list \
  --owner "$OWNER" \
  --limit "$LIMIT" \
  --format json > "$collections_json"

jq -r '.[] | select((.slug // "") | test("-caix-")) | .slug' "$collections_json" | sort -u > "$slugs"

if [[ ! -s "$slugs" ]]; then
  echo "error: no live $OWNER/*-caix-* collections found" >&2
  exit 1
fi

: > "$items"
while IFS= read -r slug; do
  [[ -z "$slug" ]] && continue
  curl -fsSL "https://huggingface.co/api/collections/$slug" \
    | jq -r --arg slug "$slug" --arg prefix "$OWNER/rhm-" '
        .items[]?
        | (.id // .item.id // .item_id // "") as $id
        | select(($id | startswith($prefix)) and ($id | endswith("-caix")))
        | [
            $slug,
            $id,
            ((.note.text // .note // "") | tostring | gsub("[\t\r\n]"; " "))
          ]
        | @tsv
      ' >> "$items"
done < "$slugs"

if [[ ! -s "$items" ]]; then
  echo "error: no $OWNER/rhm-*-caix model items found in live caix collections" >&2
  exit 1
fi

awk -F '\t' '$1 != "" && $1 != "repo" && $1 !~ /^#/ { print $1 }' "$MANIFEST" | sort -u > "$manifest_repos"
awk -F '\t' '$2 != "" { print $2 }' "$items" | sort -u > "$collection_repos"

missing="$(comm -23 "$manifest_repos" "$collection_repos")"
if [[ -n "$missing" ]]; then
  echo "error: manifest repos missing from live caix collections:" >&2
  printf '%s\n' "$missing" >&2
  exit 1
fi

bad_notes="$(
  awk -F '\t' '
    BEGIN {
      bad = "tok/s|fastest|blazing|flagship|highest quality|largest|coming soon|pending"
    }
    {
      note = tolower($3)
      if (note ~ bad || note ~ /[0-9]+([.][0-9]+)?[[:space:]]*tok\/s/) {
        printf "%s\t%s\t%s\n", $1, $2, $3
      }
    }
  ' "$items"
)"

if [[ -n "$bad_notes" ]]; then
  echo "error: public collection notes need cleanup:" >&2
  printf '%s\n' "$bad_notes" >&2
  exit 1
fi

count="$(wc -l < "$manifest_repos" | tr -d ' ')"
collections_count="$(wc -l < "$slugs" | tr -d ' ')"
echo "hf collection coverage ok: $count manifest repos covered across $collections_count collections"
