#!/usr/bin/env bash
# Compare the benchmark manifest with live RHM caix model repos on Hugging Face.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-benchmark-coverage.sh [options]

Options:
  --manifest <path>  TSV manifest. Default: benchmarks/MANIFEST.tsv.
  --author <name>    Hugging Face author. Default: redhillsmediafl.
  --search <text>    Hugging Face search term. Default: caix.
  --limit <n>        Max Hub rows to inspect. Default: 200.

Reads Hugging Face metadata only. Does not download model files.
Fails when a live redhillsmediafl/rhm-*-caix repo is missing from the manifest, or when the
manifest contains a repo that no longer appears in live caix repo discovery.
Also fails on non-canonical benchmark_mode values.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MANIFEST="$REPO_DIR/benchmarks/MANIFEST.tsv"
AUTHOR="redhillsmediafl"
SEARCH="caix"
LIMIT=200

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="${2:?}"; shift 2 ;;
    --author) AUTHOR="${2:?}"; shift 2 ;;
    --search) SEARCH="${2:?}"; shift 2 ;;
    --limit) LIMIT="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$MANIFEST" ]] || { echo "error: manifest not found: $MANIFEST" >&2; exit 2; }
[[ "$LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "error: --limit must be a positive integer" >&2; exit 2; }
command -v hf >/dev/null 2>&1 || { echo "error: hf CLI not found" >&2; exit 2; }
command -v jq >/dev/null 2>&1 || { echo "error: jq is required to parse hf JSON" >&2; exit 2; }

mode_errors="$(awk -F '\t' '
  $1 == "" || $1 == "repo" || $1 ~ /^#/ { next }
  $4 != "decode" && $4 != "speculative" && $4 != "eagle-mtp" && $4 != "manual" {
    printf "%s\t%s\n", $1, $4
  }
' "$MANIFEST")"
if [[ -n "$mode_errors" ]]; then
  echo "error: unsupported or non-canonical benchmark_mode values in $MANIFEST:" >&2
  printf '%s\n' "$mode_errors" >&2
  exit 1
fi

live_tmp="$(mktemp "${TMPDIR:-/tmp}/caix-live-repos.XXXXXX")"
manifest_tmp="$(mktemp "${TMPDIR:-/tmp}/caix-manifest-repos.XXXXXX")"
trap 'rm -f "$live_tmp" "$manifest_tmp"' EXIT

HF_HUB_DISABLE_PROGRESS_BARS=1 hf models list \
  --author "$AUTHOR" \
  --search "$SEARCH" \
  --limit "$LIMIT" \
  --format json \
  | jq -r --arg prefix "$AUTHOR/rhm-" '
      .[]
      | select((.id | startswith($prefix)) and (.id | endswith("-caix")))
      | select((.library_name == "caix") or ((.tags // []) | index("caix")))
      | .id
    ' \
  | sort -u > "$live_tmp"

awk -F '\t' '$1 != "" && $1 != "repo" && $1 !~ /^#/ { print $1 }' "$MANIFEST" | sort -u > "$manifest_tmp"

if [[ ! -s "$live_tmp" ]]; then
  echo "error: no live $AUTHOR/rhm-*-caix repos found via hf models list" >&2
  exit 1
fi

missing="$(comm -23 "$live_tmp" "$manifest_tmp")"
extra="$(comm -13 "$live_tmp" "$manifest_tmp")"

if [[ -n "$missing" ]]; then
  echo "error: live caix repos missing from $MANIFEST:" >&2
  printf '%s\n' "$missing" >&2
fi
if [[ -n "$extra" ]]; then
  echo "error: manifest repos not found in live caix discovery:" >&2
  printf '%s\n' "$extra" >&2
fi
if [[ -n "$missing" || -n "$extra" ]]; then
  exit 1
fi

count="$(wc -l < "$manifest_tmp" | tr -d ' ')"
echo "benchmark coverage ok: $count live repos covered"
