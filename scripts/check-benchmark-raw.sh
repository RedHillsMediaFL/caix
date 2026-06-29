#!/usr/bin/env bash
# Check committed raw benchmark evidence for comparability and reproducibility.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-benchmark-raw.sh [options]

Options:
  --raw-dir <path>  Raw benchmark root. Default: benchmarks/raw.

Does not run models, download payloads, or contact Hugging Face.
Fails when committed suite/model raw logs are missing metadata, new or changed raw dirs have dirty
run-start git status, measured rows fail, pinned model repo revisions are missing, suite/model
settings drift, or deterministic measured stdout differs.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="$REPO_DIR/benchmarks/raw"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw-dir) RAW_DIR="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -d "$RAW_DIR" ]] || { echo "error: raw benchmark directory not found: $RAW_DIR" >&2; exit 2; }

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

canonical_benchmark_mode() {
  case "$1" in
    eagle) printf 'eagle-mtp' ;;
    "") printf '-' ;;
    *) printf '%s' "$1" ;;
  esac
}

repo_relative_path() {
  local path="$1"
  local abs
  abs="$(cd "$path" && pwd)"
  case "$abs/" in
    "$REPO_DIR"/*)
      printf '%s\n' "${abs#$REPO_DIR/}"
      ;;
    *)
      return 1
      ;;
  esac
}

raw_dir_has_git_changes() {
  local raw_dir="$1"
  local rel
  rel="$(repo_relative_path "$raw_dir")" || return 0
  [[ -n "$(git -C "$REPO_DIR" status --porcelain -- "$rel")" ]]
}

summary_count() {
  local file="$1"
  local expr="$2"
  awk -F '\t' "$expr" "$file"
}

check_model_dir() {
  local suite="$1"
  local repo="$2"
  local row_mode="$3"
  local output="$4"

  [[ -d "$output" ]] || { echo "error: raw output directory missing for $repo: $output" >&2; return 1; }
  [[ -f "$output/metadata.txt" ]] || { echo "error: raw metadata missing for $repo: $output/metadata.txt" >&2; return 1; }
  [[ -f "$output/summary.tsv" ]] || { echo "error: raw summary missing for $repo: $output/summary.tsv" >&2; return 1; }

  local metadata_repo
  metadata_repo="$(metadata_value repo "$output/metadata.txt")"
  if [[ "$metadata_repo" != "$repo" ]]; then
    echo "error: metadata repo mismatch for $repo: $metadata_repo" >&2
    return 1
  fi

  local repo_revision
  repo_revision="$(metadata_value repo_revision "$output/metadata.txt")"
  if [[ ! "$repo_revision" =~ ^[0-9a-f]{40}$ ]]; then
    echo "error: missing pinned repo revision for $repo: $repo_revision" >&2
    return 1
  fi

  if raw_dir_has_git_changes "$output"; then
    local git_status
    git_status="$(metadata_value git_status "$output/metadata.txt")"
    if [[ "$git_status" != "0 dirty entries" ]]; then
      echo "error: dirty run-start git status for $repo: $git_status" >&2
      return 1
    fi
  fi

  local runs measured failed
  runs="$(metadata_value runs "$output/metadata.txt")"
  [[ "$runs" =~ ^[1-9][0-9]*$ ]] || { echo "error: invalid runs metadata for $repo: $runs" >&2; return 1; }
  measured="$(summary_count "$output/summary.tsv" 'NR > 1 && $1 == "measured" && $3 == "ok" { n++ } END { print n + 0 }')"
  failed="$(summary_count "$output/summary.tsv" 'NR > 1 && $1 == "measured" && $3 != "ok" { n++ } END { print n + 0 }')"
  if [[ "$failed" != "0" ]]; then
    echo "error: measured failures for $repo: $failed" >&2
    return 1
  fi
  if [[ "$measured" != "$runs" ]]; then
    echo "error: measured run count mismatch for $repo: summary=$measured metadata=$runs" >&2
    return 1
  fi

  local generated_kinds
  generated_kinds="$(awk -F '\t' 'NR > 1 && $1 == "measured" && $3 == "ok" { seen[$4] = 1 } END { for (value in seen) n++; print n + 0 }' "$output/summary.tsv")"
  if [[ "$generated_kinds" != "1" ]]; then
    echo "error: measured generated-token count is not stable for $repo" >&2
    return 1
  fi

  local first_stdout=""
  local stdout_path
  while IFS=$'\t' read -r _phase _run _status _generated _load _prefill _decode _tps stdout_path _stderr; do
    [[ "$_phase" == "measured" && "$_status" == "ok" ]] || continue
    [[ -f "$stdout_path" ]] || { echo "error: measured stdout missing for $repo: $stdout_path" >&2; return 1; }
    if [[ -z "$first_stdout" ]]; then
      first_stdout="$stdout_path"
    elif ! cmp -s "$first_stdout" "$stdout_path"; then
      echo "error: measured stdout differs for $repo: $first_stdout vs $stdout_path" >&2
      return 1
    fi
  done < "$output/summary.tsv"

  local suite_prompt suite_max_tokens suite_temperature suite_raw
  suite_prompt="$(metadata_value prompt "$suite/metadata.txt")"
  suite_max_tokens="$(metadata_value max_tokens "$suite/metadata.txt")"
  suite_temperature="$(metadata_value temperature "$suite/metadata.txt")"
  suite_raw="$(metadata_value raw "$suite/metadata.txt")"

  local model_prompt model_max_tokens model_temperature model_raw model_mode
  model_prompt="$(metadata_value prompt "$output/metadata.txt")"
  model_max_tokens="$(metadata_value max_tokens "$output/metadata.txt")"
  model_temperature="$(metadata_value temperature "$output/metadata.txt")"
  model_raw="$(metadata_value raw "$output/metadata.txt")"
  model_mode="$(canonical_benchmark_mode "$(metadata_value benchmark_mode "$output/metadata.txt")")"
  row_mode="$(canonical_benchmark_mode "$row_mode")"

  if [[ "$model_prompt" != "$suite_prompt" || "$model_max_tokens" != "$suite_max_tokens" ||
        "$model_temperature" != "$suite_temperature" || "$model_raw" != "$suite_raw" ]]; then
    echo "error: suite/model benchmark settings drift for $repo" >&2
    return 1
  fi
  if [[ "$model_mode" != "$row_mode" ]]; then
    echo "error: benchmark mode mismatch for $repo: suite=$row_mode model=$model_mode" >&2
    return 1
  fi
}

suite_count=0
model_count=0

while IFS= read -r suite; do
  [[ -f "$suite/metadata.txt" ]] || { echo "error: suite metadata missing: $suite/metadata.txt" >&2; exit 1; }
  [[ -f "$suite/summary.tsv" ]] || { echo "error: suite summary missing: $suite/summary.tsv" >&2; exit 1; }
  suite_count=$((suite_count + 1))

  total="$(metadata_value total "$suite/metadata.txt")"
  measured_meta="$(metadata_value measured "$suite/metadata.txt")"
  skipped_meta="$(metadata_value skipped "$suite/metadata.txt")"
  failed_meta="$(metadata_value failed "$suite/metadata.txt")"
  [[ "$total" =~ ^[0-9]+$ && "$measured_meta" =~ ^[0-9]+$ && "$skipped_meta" =~ ^[0-9]+$ && "$failed_meta" =~ ^[0-9]+$ ]] || {
    echo "error: invalid suite counts in $suite/metadata.txt" >&2
    exit 1
  }

  if raw_dir_has_git_changes "$suite"; then
    suite_git_status="$(metadata_value git_status "$suite/metadata.txt")"
    if [[ "$suite_git_status" != "0 dirty entries" ]]; then
      echo "error: dirty run-start git status in suite $suite: $suite_git_status" >&2
      exit 1
    fi
  fi

  read -r rows measured_rows skipped_rows failed_rows < <(
    awk -F '\t' '
      NR > 1 {
        rows++
        if ($5 == "measured") measured++
        else if ($5 == "skip") skipped++
        else if ($5 == "fail") failed++
      }
      END { printf "%d %d %d %d\n", rows + 0, measured + 0, skipped + 0, failed + 0 }
    ' "$suite/summary.tsv"
  )
  if [[ "$rows" != "$total" || "$measured_rows" != "$measured_meta" ||
        "$skipped_rows" != "$skipped_meta" || "$failed_rows" != "$failed_meta" ]]; then
    echo "error: suite count mismatch in $suite" >&2
    exit 1
  fi

  while IFS=$'\t' read -r repo _local_dir _kind benchmark_mode status _reason _bundle output; do
    [[ -z "${repo:-}" || "$repo" == "repo" || "$repo" == \#* ]] && continue
    [[ "$status" == "measured" ]] || continue
    check_model_dir "$suite" "$repo" "$benchmark_mode" "$output"
    model_count=$((model_count + 1))
  done < "$suite/summary.tsv"
done < <(find "$RAW_DIR" -maxdepth 1 -type d -name '*-suite' | sort)

[[ "$suite_count" -gt 0 ]] || { echo "error: no suite raw directories found under $RAW_DIR" >&2; exit 1; }
[[ "$model_count" -gt 0 ]] || { echo "error: no measured model raw directories found under $RAW_DIR" >&2; exit 1; }

echo "benchmark raw logs ok: $suite_count suites, $model_count measured model runs checked"
