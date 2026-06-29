#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/collect-model-revisions.sh [options]

Options:
  --manifest <path>  TSV manifest. Default: benchmarks/MANIFEST.tsv.
  --out <path>       Output repo<TAB>revision TSV. Default: benchmarks/revisions.tsv.
  --details <path>   Optional audit TSV with Hub metadata.
  --quiet            Do not print per-repo progress.

Reads model metadata from Hugging Face. Does not download model files.
Fails without writing the output if any manifest repo is missing a commit SHA.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MANIFEST="$REPO_DIR/benchmarks/MANIFEST.tsv"
OUT="$REPO_DIR/benchmarks/revisions.tsv"
DETAILS=""
QUIET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="${2:?}"; shift 2 ;;
    --out) OUT="${2:?}"; shift 2 ;;
    --details) DETAILS="${2:?}"; shift 2 ;;
    --quiet) QUIET=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$MANIFEST" ]] || { echo "error: manifest not found: $MANIFEST" >&2; exit 2; }
command -v hf >/dev/null 2>&1 || { echo "error: hf CLI not found" >&2; exit 2; }
command -v jq >/dev/null 2>&1 || { echo "error: jq is required to parse hf JSON" >&2; exit 2; }

mkdir -p "$(dirname "$OUT")"
OUT_TMP="$OUT.tmp.$$"
DETAILS_TMP=""
if [[ -n "$DETAILS" ]]; then
  mkdir -p "$(dirname "$DETAILS")"
  DETAILS_TMP="$DETAILS.tmp.$$"
fi
trap 'rm -f "$OUT_TMP" "$DETAILS_TMP"' EXIT

: > "$OUT_TMP"
if [[ -n "$DETAILS_TMP" ]]; then
  printf 'repo\trevision\tlast_modified\tgated\tprivate\tlibrary\tstorage_bytes\tlicense\tlocal_dir\tkind\tbenchmark_mode\tstatus\n' > "$DETAILS_TMP"
fi

json_field() {
  local jq_expr="$1"
  local input="$2"
  jq -r "$jq_expr" <<< "$input" | tr '\t\r\n' '   ' | sed -E 's/[[:space:]]+$//'
}

total=0
written=0

while IFS=$'\t' read -r repo local_dir kind benchmark_mode status notes; do
  [[ -z "${repo:-}" || "$repo" == "repo" || "$repo" == \#* ]] && continue
  total=$((total + 1))

  [[ "$QUIET" == "1" ]] || echo "metadata: $repo" >&2
  if ! info="$(HF_HUB_DISABLE_PROGRESS_BARS=1 hf models info "$repo" --format json)"; then
    echo "error: hf models info failed for $repo" >&2
    exit 1
  fi

  model_id="$(json_field '.id // ""' "$info")"
  if [[ "$model_id" != "$repo" ]]; then
    echo "error: Hub returned '$model_id' while resolving '$repo'" >&2
    exit 1
  fi

  revision="$(json_field '.sha // ""' "$info")"
  if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]; then
    echo "error: missing or invalid commit SHA for $repo" >&2
    exit 1
  fi

  printf '%s\t%s\n' "$repo" "$revision" >> "$OUT_TMP"

  if [[ -n "$DETAILS_TMP" ]]; then
    last_modified="$(json_field '.last_modified // "-"' "$info")"
    gated="$(json_field '(if has("gated") and .gated != null then .gated else "-" end) | tostring' "$info")"
    private="$(json_field '(if has("private") and .private != null then .private else "-" end) | tostring' "$info")"
    library="$(json_field '.library_name // "-"' "$info")"
    storage="$(json_field '(.used_storage // "-") | tostring' "$info")"
    license="$(json_field '([.tags[]? | select(startswith("license:")) | sub("^license:"; "")] | first) // "-"' "$info")"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$repo" "$revision" "$last_modified" "$gated" "$private" "$library" "$storage" "$license" \
      "$local_dir" "$kind" "$benchmark_mode" "$status" >> "$DETAILS_TMP"
  fi

  written=$((written + 1))
done < "$MANIFEST"

[[ "$written" -gt 0 ]] || { echo "error: no model repos found in $MANIFEST" >&2; exit 1; }

mv "$OUT_TMP" "$OUT"
if [[ -n "$DETAILS_TMP" ]]; then
  mv "$DETAILS_TMP" "$DETAILS"
fi
trap - EXIT

echo "$OUT"
if [[ -n "$DETAILS" ]]; then
  echo "$DETAILS"
fi
echo "recorded $written of $total manifest repos" >&2
