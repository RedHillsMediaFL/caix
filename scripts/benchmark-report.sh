#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/benchmark-report.sh --suite <suite-dir> [--out <path>]

Options:
  --suite <dir>  Suite output directory from scripts/benchmark-suite.sh.
  --out <path>   Report path. Default: <suite-dir>/report.tsv.

Reads suite summary rows plus per-model raw benchmark summaries.
Refuses measured rows with missing raw logs or failed measured runs.
Rows without a recorded model revision are marked publishable=no.
USAGE
}

SUITE=""
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite) SUITE="${2:?}"; shift 2 ;;
    --out) OUT="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$SUITE" ]] || { echo "error: --suite is required" >&2; exit 2; }
[[ -d "$SUITE" ]] || { echo "error: suite directory not found: $SUITE" >&2; exit 2; }
[[ -f "$SUITE/summary.tsv" ]] || { echo "error: suite summary not found: $SUITE/summary.tsv" >&2; exit 2; }
[[ -f "$SUITE/metadata.txt" ]] || { echo "error: suite metadata not found: $SUITE/metadata.txt" >&2; exit 2; }

if [[ -z "$OUT" ]]; then
  OUT="$SUITE/report.tsv"
fi
mkdir -p "$(dirname "$OUT")"
TMP="$OUT.tmp.$$"
trap 'rm -f "$TMP"' EXIT

metadata_value() {
  local key="$1"
  local file="$2"
  awk -v key="$key" '
    index($0, key "=") == 1 {
      sub("^[^=]*=", "")
      print
      exit
    }
  ' "$file"
}

count_measured_ok() {
  awk -F '\t' 'NR > 1 && $1 == "measured" && $3 == "ok" && $8 ~ /^[0-9]+([.][0-9]+)?$/ { n++ } END { print n + 0 }' "$1"
}

count_measured_failed() {
  awk -F '\t' 'NR > 1 && $1 == "measured" && $3 != "ok" { n++ } END { print n + 0 }' "$1"
}

median_field() {
  local file="$1"
  local field="$2"
  awk -F '\t' -v field="$field" \
    'NR > 1 && $1 == "measured" && $3 == "ok" && $field ~ /^[0-9]+([.][0-9]+)?$/ { print $field }' "$file" \
    | sort -n \
    | awk '
        { values[NR] = $1 }
        END {
          if (NR == 0) {
            print "-"
          } else if (NR % 2 == 1) {
            printf "%.6g\n", values[(NR + 1) / 2]
          } else {
            printf "%.6g\n", (values[NR / 2] + values[NR / 2 + 1]) / 2
          }
        }'
}

min_field() {
  local file="$1"
  local field="$2"
  awk -F '\t' -v field="$field" \
    'NR > 1 && $1 == "measured" && $3 == "ok" && $field ~ /^[0-9]+([.][0-9]+)?$/ { print $field }' "$file" \
    | sort -n \
    | awk 'NR == 1 { print; found = 1 } END { if (!found) print "-" }'
}

max_field() {
  local file="$1"
  local field="$2"
  awk -F '\t' -v field="$field" \
    'NR > 1 && $1 == "measured" && $3 == "ok" && $field ~ /^[0-9]+([.][0-9]+)?$/ { print $field }' "$file" \
    | sort -n \
    | awk '{ value = $1; found = 1 } END { if (found) print value; else print "-" }'
}

suite_caix_commit="$(metadata_value caix_commit "$SUITE/metadata.txt")"
suite_machine="$(metadata_value machine "$SUITE/metadata.txt")"
suite_memory="$(metadata_value memory_bytes "$SUITE/metadata.txt")"
suite_os="$(metadata_value os "$SUITE/metadata.txt")"
suite_prompt="$(metadata_value prompt "$SUITE/metadata.txt")"
suite_max_tokens="$(metadata_value max_tokens "$SUITE/metadata.txt")"
suite_temperature="$(metadata_value temperature "$SUITE/metadata.txt")"
suite_raw="$(metadata_value raw "$SUITE/metadata.txt")"

printf 'repo\trepo_revision\tlocal_dir\tkind\tstatus\tpublishable\treason\tmeasured_runs\tmedian_generated\tmedian_load_s\tmedian_prefill_s\tmedian_decode_s\tmedian_decode_tps\tmin_decode_tps\tmax_decode_tps\tcaix_commit\tmachine\tmemory_bytes\tos\tmax_tokens\ttemperature\traw\tprompt\traw_dir\n' > "$TMP"

while IFS=$'\t' read -r repo local_dir kind status reason bundle output; do
  [[ -z "${repo:-}" || "$repo" == "repo" || "$repo" == \#* ]] && continue

  repo_revision="-"
  publishable="no"
  measured_runs="-"
  median_generated="-"
  median_load="-"
  median_prefill="-"
  median_decode="-"
  median_tps="-"
  min_tps="-"
  max_tps="-"
  caix_commit="$suite_caix_commit"
  machine="$suite_machine"
  memory="$suite_memory"
  os="$suite_os"
  max_tokens="$suite_max_tokens"
  temperature="$suite_temperature"
  raw="$suite_raw"
  prompt="$suite_prompt"
  raw_dir="$output"

  if [[ "$status" == "measured" ]]; then
    [[ -d "$output" ]] || { echo "error: raw output directory missing for $repo: $output" >&2; exit 1; }
    [[ -f "$output/summary.tsv" ]] || { echo "error: raw summary missing for $repo: $output/summary.tsv" >&2; exit 1; }
    [[ -f "$output/metadata.txt" ]] || { echo "error: raw metadata missing for $repo: $output/metadata.txt" >&2; exit 1; }

    failed="$(count_measured_failed "$output/summary.tsv")"
    [[ "$failed" == "0" ]] || { echo "error: measured run failed for $repo ($failed failed rows)" >&2; exit 1; }

    measured_runs="$(count_measured_ok "$output/summary.tsv")"
    [[ "$measured_runs" -gt 0 ]] || { echo "error: no measured ok rows for $repo" >&2; exit 1; }

    metadata_repo="$(metadata_value repo "$output/metadata.txt")"
    [[ -z "$metadata_repo" || "$metadata_repo" == "$repo" ]] || {
      echo "error: metadata repo mismatch for $repo: $metadata_repo" >&2
      exit 1
    }
    repo_revision="$(metadata_value repo_revision "$output/metadata.txt")"
    [[ -n "$repo_revision" ]] || repo_revision="unknown"

    caix_commit="$(metadata_value caix_commit "$output/metadata.txt")"
    machine="$(metadata_value machine "$output/metadata.txt")"
    memory="$(metadata_value memory_bytes "$output/metadata.txt")"
    os="$(metadata_value os "$output/metadata.txt")"
    max_tokens="$(metadata_value max_tokens "$output/metadata.txt")"
    temperature="$(metadata_value temperature "$output/metadata.txt")"
    raw="$(metadata_value raw "$output/metadata.txt")"
    prompt="$(metadata_value prompt "$output/metadata.txt")"

    median_generated="$(median_field "$output/summary.tsv" 4)"
    median_load="$(median_field "$output/summary.tsv" 5)"
    median_prefill="$(median_field "$output/summary.tsv" 6)"
    median_decode="$(median_field "$output/summary.tsv" 7)"
    median_tps="$(median_field "$output/summary.tsv" 8)"
    min_tps="$(min_field "$output/summary.tsv" 8)"
    max_tps="$(max_field "$output/summary.tsv" 8)"

    if [[ "$repo_revision" == "unknown" ]]; then
      reason="missing repo revision in raw metadata"
    else
      reason="-"
      publishable="yes"
    fi
  elif [[ "$status" == "planned" ]]; then
    reason="${reason:-dry run}"
  elif [[ "$status" == "skip" || "$status" == "fail" ]]; then
    reason="${reason:--}"
  else
    echo "error: unknown suite status for $repo: $status" >&2
    exit 1
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$repo" "$repo_revision" "$local_dir" "$kind" "$status" "$publishable" "$reason" \
    "$measured_runs" "$median_generated" "$median_load" "$median_prefill" "$median_decode" \
    "$median_tps" "$min_tps" "$max_tps" "$caix_commit" "$machine" "$memory" "$os" \
    "$max_tokens" "$temperature" "$raw" "$prompt" "$raw_dir" >> "$TMP"
done < "$SUITE/summary.tsv"

mv "$TMP" "$OUT"
trap - EXIT
echo "$OUT"
