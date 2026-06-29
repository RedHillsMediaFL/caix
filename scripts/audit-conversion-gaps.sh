#!/usr/bin/env bash
# Emit a metadata-only audit for active registry conversion lanes.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/audit-conversion-gaps.sh [options]

Options:
  --registry <path>  Model registry JSON. Default: models/registry.json.
  --ledger <path>    Conversion ledger TSV. Default: docs/CONVERSION_LEDGER.tsv.
  --manifest <path>  Benchmark manifest TSV. Default: benchmarks/MANIFEST.tsv.
  --out <path>       Output TSV path. Default: stdout.

Reads Hugging Face model metadata only. Does not download model payloads.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

REGISTRY="$REPO_DIR/models/registry.json"
LEDGER="$REPO_DIR/docs/CONVERSION_LEDGER.tsv"
MANIFEST="$REPO_DIR/benchmarks/MANIFEST.tsv"
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="${2:?}"; shift 2 ;;
    --ledger) LEDGER="${2:?}"; shift 2 ;;
    --manifest) MANIFEST="${2:?}"; shift 2 ;;
    --out) OUT="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$REGISTRY" ]] || { echo "error: registry not found: $REGISTRY" >&2; exit 2; }
[[ -f "$LEDGER" ]] || { echo "error: ledger not found: $LEDGER" >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "error: manifest not found: $MANIFEST" >&2; exit 2; }
command -v hf >/dev/null 2>&1 || { echo "error: hf CLI not found" >&2; exit 2; }
command -v jq >/dev/null 2>&1 || { echo "error: jq is required" >&2; exit 2; }

export HF_HOME="${HF_HOME:-/Volumes/SSD/hf-cache}"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-conversion-audit.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

registry_tsv="$tmpdir/registry.tsv"
manifest_repos="$tmpdir/manifest-repos.txt"
audit_tmp="$tmpdir/audit.tsv"

jq -e '
  .models as $models
  | (.conversion_order | type == "array")
  and all(.conversion_order[]; ($models[.] and (($models[.].hf_repo // "") != "")))
' "$REGISTRY" >/dev/null

jq -r '
  .conversion_order[] as $key
  | .models[$key] as $model
  | [
      $key,
      $model.hf_repo,
      ($model.model_type // "-"),
      ($model.op_family // "-"),
      ($model.role // "-"),
      (($model.params_b // "-") | tostring),
      (($model.bf16_gb // "-") | tostring),
      ($model.compression // "-"),
      (($model.downloaded // false) | tostring)
    ]
  | @tsv
' "$REGISTRY" > "$registry_tsv"

awk -F '\t' '
  NR == 1 {
    expected = "model_key\tsource_repo\tstatus\tpublished_repo\tnext_step"
    if ($0 != expected) {
      printf "error: bad ledger header: %s\n", $0 > "/dev/stderr"
      exit 2
    }
    next
  }
  NF != 5 {
    printf "error: bad ledger row %d: expected 5 tab-separated fields\n", NR > "/dev/stderr"
    exit 2
  }
  { print }
' "$LEDGER" > "$tmpdir/ledger-body.tsv"

awk -F '\t' '$1 != "" && $1 != "repo" && $1 !~ /^#/ { print $1 }' "$MANIFEST" \
  | sort -u > "$manifest_repos"

json_field() {
  local jq_expr="$1"
  local input="$2"
  jq -r "$jq_expr" <<< "$input" | tr '\t\r\n' '   ' | sed -E 's/[[:space:]]+$//'
}

manifest_status() {
  local published="$1"
  if [[ "$published" == "-" ]]; then
    printf 'not_published'
    return
  fi

  local missing=()
  local repo
  IFS=',' read -ra repos <<< "$published"
  for repo in "${repos[@]}"; do
    if ! grep -Fxq "$repo" "$manifest_repos"; then
      missing+=("$repo")
    fi
  done

  if [[ "${#missing[@]}" -eq 0 ]]; then
    printf 'present'
  else
    local IFS=','
    printf 'missing:%s' "${missing[*]}"
  fi
}

printf 'model_key\tsource_repo\tsource_revision\tlast_modified\tgated\tprivate\tlibrary\tstorage_bytes\tlicense\tmodel_type\top_family\trole\tparams_b\tbf16_gb\tcompression\tdownloaded\tledger_status\tpublished_repo\tmanifest_status\tnext_step\n' > "$audit_tmp"

while IFS=$'\t' read -r key source_repo model_type op_family role params_b bf16_gb compression downloaded; do
  ledger_row="$(awk -F '\t' -v key="$key" '$1 == key { print; found = 1 } END { exit found ? 0 : 1 }' "$tmpdir/ledger-body.tsv")" || {
    echo "error: registry key missing from ledger: $key" >&2
    exit 1
  }

  IFS=$'\t' read -r ledger_key ledger_source ledger_status published_repo next_step <<< "$ledger_row"
  if [[ "$ledger_source" != "$source_repo" ]]; then
    echo "error: source repo mismatch for $key: ledger=$ledger_source registry=$source_repo" >&2
    exit 1
  fi

  status="$(manifest_status "$published_repo")"
  if [[ "$published_repo" != "-" && "$status" != "present" ]]; then
    echo "error: published repo missing from manifest for $key: $status" >&2
    exit 1
  fi

  if ! info="$(HF_HUB_DISABLE_PROGRESS_BARS=1 hf models info "$source_repo" --format json)"; then
    echo "error: hf models info failed for $source_repo" >&2
    exit 1
  fi

  model_id="$(json_field '.id // ""' "$info")"
  if [[ "$model_id" != "$source_repo" ]]; then
    echo "error: Hub returned '$model_id' while resolving '$source_repo'" >&2
    exit 1
  fi

  revision="$(json_field '.sha // ""' "$info")"
  if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]]; then
    echo "error: missing or invalid source commit SHA for $source_repo" >&2
    exit 1
  fi

  last_modified="$(json_field '.last_modified // "-"' "$info")"
  gated="$(json_field '(if has("gated") and .gated != null then .gated else "-" end) | tostring' "$info")"
  private="$(json_field '(if has("private") and .private != null then .private else "-" end) | tostring' "$info")"
  library="$(json_field '.library_name // "-"' "$info")"
  storage="$(json_field '(.used_storage // "-") | tostring' "$info")"
  license="$(json_field '([.tags[]? | select(startswith("license:")) | sub("^license:"; "")] | first) // "-"' "$info")"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$key" "$source_repo" "$revision" "$last_modified" "$gated" "$private" "$library" \
    "$storage" "$license" "$model_type" "$op_family" "$role" "$params_b" "$bf16_gb" \
    "$compression" "$downloaded" "$ledger_status" "$published_repo" "$status" "$next_step" \
    >> "$audit_tmp"
done < "$registry_tsv"

if [[ -n "$OUT" ]]; then
  mkdir -p "$(dirname "$OUT")"
  mv "$audit_tmp" "$OUT"
  echo "$OUT"
else
  cat "$audit_tmp"
fi
