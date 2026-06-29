#!/usr/bin/env bash
# Check that the committed conversion gap audit matches local registry and ledger state.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-conversion-gap-audit.sh [options]

Options:
  --registry <path>  Model registry JSON. Default: models/registry.json.
  --ledger <path>    Conversion ledger TSV. Default: docs/CONVERSION_LEDGER.tsv.
  --manifest <path>  Benchmark manifest TSV. Default: benchmarks/MANIFEST.tsv.
  --audit <path>     Conversion gap audit TSV. Default: docs/CONVERSION_GAP_AUDIT.tsv.

Does not use Hugging Face and does not download models. Fails when the committed audit no longer
matches the registry, conversion ledger, or benchmark manifest.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

REGISTRY="$REPO_DIR/models/registry.json"
LEDGER="$REPO_DIR/docs/CONVERSION_LEDGER.tsv"
MANIFEST="$REPO_DIR/benchmarks/MANIFEST.tsv"
AUDIT="$REPO_DIR/docs/CONVERSION_GAP_AUDIT.tsv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="${2:?}"; shift 2 ;;
    --ledger) LEDGER="${2:?}"; shift 2 ;;
    --manifest) MANIFEST="${2:?}"; shift 2 ;;
    --audit) AUDIT="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$REGISTRY" ]] || { echo "error: registry not found: $REGISTRY" >&2; exit 2; }
[[ -f "$LEDGER" ]] || { echo "error: ledger not found: $LEDGER" >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "error: manifest not found: $MANIFEST" >&2; exit 2; }
[[ -f "$AUDIT" ]] || { echo "error: audit not found: $AUDIT" >&2; exit 2; }
command -v jq >/dev/null 2>&1 || { echo "error: jq is required" >&2; exit 2; }

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-conversion-gap-check.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

registry_tsv="$tmpdir/registry.tsv"
ledger_body="$tmpdir/ledger-body.tsv"
manifest_repos="$tmpdir/manifest-repos.txt"

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
  $1 == "" || $2 == "" || $3 == "" || $4 == "" || $5 == "" {
    printf "error: empty ledger field on row %d\n", NR > "/dev/stderr"
    exit 2
  }
  { print }
' "$LEDGER" > "$ledger_body"

awk -F '\t' '$1 != "" && $1 != "repo" && $1 !~ /^#/ { print $1 }' "$MANIFEST" \
  | sort -u > "$manifest_repos"

awk -F '\t' -v registry="$registry_tsv" -v ledger="$ledger_body" -v manifest="$manifest_repos" '
  BEGIN {
    header = "model_key\tsource_repo\tsource_revision\tlast_modified\tgated\tprivate\tlibrary\tstorage_bytes\tlicense\tmodel_type\top_family\trole\tparams_b\tbf16_gb\tcompression\tdownloaded\tledger_status\tpublished_repo\tmanifest_status\tnext_step"
    while ((getline line < registry) > 0) {
      split(line, parts, "\t")
      key = parts[1]
      order[++expected_count] = key
      source[key] = parts[2]
      model_type[key] = parts[3]
      op_family[key] = parts[4]
      role[key] = parts[5]
      params_b[key] = parts[6]
      bf16_gb[key] = parts[7]
      compression[key] = parts[8]
      downloaded[key] = parts[9]
    }
    close(registry)

    while ((getline line < ledger) > 0) {
      split(line, parts, "\t")
      key = parts[1]
      ledger_source[key] = parts[2]
      ledger_status[key] = parts[3]
      published_repo[key] = parts[4]
      next_step[key] = parts[5]
    }
    close(ledger)

    while ((getline repo < manifest) > 0) {
      manifest_repo[repo] = 1
    }
    close(manifest)
  }
  NR == 1 {
    if ($0 != header) {
      printf "error: bad audit header: %s\n", $0 > "/dev/stderr"
      fail = 1
    }
    next
  }
  {
    if (NF != 20) {
      printf "error: bad audit row %d: expected 20 tab-separated fields, got %d\n", NR, NF > "/dev/stderr"
      fail = 1
      next
    }
    for (i = 1; i <= NF; i++) {
      if ($i == "") {
        printf "error: empty audit field row %d column %d\n", NR, i > "/dev/stderr"
        fail = 1
      }
    }

    row += 1
    key = $1
    if (key != order[row]) {
      printf "error: audit row order mismatch at row %d: audit=%s registry=%s\n", NR, key, order[row] > "/dev/stderr"
      fail = 1
    }
    if (seen[key]++) {
      printf "error: duplicate audit row for %s\n", key > "/dev/stderr"
      fail = 1
    }
    if (!(key in source)) {
      printf "error: audit row not present in registry conversion_order: %s\n", key > "/dev/stderr"
      fail = 1
      next
    }
    if (!(key in ledger_status)) {
      printf "error: audit row missing from conversion ledger: %s\n", key > "/dev/stderr"
      fail = 1
    }

    if ($2 != source[key] || $2 != ledger_source[key]) {
      printf "error: source repo mismatch for %s\n", key > "/dev/stderr"
      fail = 1
    }
    if ($3 !~ /^[0-9a-f]{40}$/) {
      printf "error: invalid source revision for %s: %s\n", key, $3 > "/dev/stderr"
      fail = 1
    }
    if ($4 != "-" && $4 !~ /^[0-9]{4}-[0-9]{2}-[0-9]{2}T/) {
      printf "error: invalid last_modified for %s: %s\n", key, $4 > "/dev/stderr"
      fail = 1
    }
    if ($5 != "-" && $5 != "false" && $5 != "true" && $5 != "auto" && $5 != "manual") {
      printf "error: invalid gated value for %s: %s\n", key, $5 > "/dev/stderr"
      fail = 1
    }
    if ($6 != "-" && $6 != "false" && $6 != "true") {
      printf "error: invalid private value for %s: %s\n", key, $6 > "/dev/stderr"
      fail = 1
    }
    if ($8 != "-" && $8 !~ /^[0-9]+$/) {
      printf "error: invalid storage_bytes for %s: %s\n", key, $8 > "/dev/stderr"
      fail = 1
    }

    if ($10 != model_type[key] || $11 != op_family[key] || $12 != role[key] ||
        $13 != params_b[key] || $14 != bf16_gb[key] || $15 != compression[key] ||
        $16 != downloaded[key]) {
      printf "error: registry metadata mismatch for %s\n", key > "/dev/stderr"
      fail = 1
    }
    if ($17 != ledger_status[key] || $18 != published_repo[key] || $20 != next_step[key]) {
      printf "error: ledger metadata mismatch for %s\n", key > "/dev/stderr"
      fail = 1
    }

    expected_manifest = (published_repo[key] == "-") ? "not_published" : "present"
    if (published_repo[key] != "-") {
      n = split(published_repo[key], repos, ",")
      for (i = 1; i <= n; i++) {
        if (!(repos[i] in manifest_repo)) {
          printf "error: published repo missing from manifest for %s: %s\n", key, repos[i] > "/dev/stderr"
          fail = 1
          expected_manifest = "missing"
        }
      }
    }
    if ($19 != expected_manifest) {
      printf "error: manifest status mismatch for %s: audit=%s expected=%s\n", key, $19, expected_manifest > "/dev/stderr"
      fail = 1
    }
  }
  END {
    if (row != expected_count) {
      printf "error: audit row count mismatch: audit=%d registry=%d\n", row, expected_count > "/dev/stderr"
      fail = 1
    }
    for (i = 1; i <= expected_count; i++) {
      key = order[i]
      if (!(key in seen)) {
        printf "error: registry conversion lane missing from audit: %s\n", key > "/dev/stderr"
        fail = 1
      }
    }
    exit fail ? 1 : 0
  }
' "$AUDIT"

count="$(($(wc -l < "$AUDIT" | tr -d ' ') - 1))"
echo "conversion gap audit ok: $count registry conversion lanes checked"
