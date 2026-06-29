#!/usr/bin/env bash
# Check committed raw benchmark evidence for comparability and reproducibility.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-benchmark-raw.sh [options]

Options:
  --raw-dir <path>     Raw benchmark root. Default: benchmarks/raw.
  --require-tracked    Require every suite/model raw evidence file to be tracked by git.

Does not run models, download payloads, or contact Hugging Face.
Fails when committed suite/model raw logs are missing metadata, new or changed raw dirs have dirty
run-start git status, measured rows fail, pinned model repo revisions are missing, caix commits are
not present in the repository, raw outputs point outside the raw evidence tree, suite/model settings
drift, or deterministic measured stdout differs.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RAW_DIR="$REPO_DIR/benchmarks/raw"
REPO_REAL="$(cd "$REPO_DIR" && pwd -P)"
REQUIRE_TRACKED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --raw-dir) RAW_DIR="${2:?}"; shift 2 ;;
    --require-tracked) REQUIRE_TRACKED=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -d "$RAW_DIR" ]] || { echo "error: raw benchmark directory not found: $RAW_DIR" >&2; exit 2; }
RAW_REAL="$(cd "$RAW_DIR" && pwd -P)"

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

resolve_local_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$REPO_DIR" "$1" ;;
  esac
}

canonical_existing_path() {
  local path="$1"
  local abs dir base

  abs="$(resolve_local_path "$path")"
  if [[ -d "$abs" ]]; then
    (cd "$abs" && pwd -P)
  else
    dir="$(cd "$(dirname "$abs")" && pwd -P)" || return 1
    base="$(basename "$abs")"
    printf '%s/%s\n' "$dir" "$base"
  fi
}

require_path_under_dir() {
  local label="$1"
  local path="$2"
  local parent="$3"
  local path_real

  path_real="$(canonical_existing_path "$path")" || {
    echo "error: cannot resolve $label path: $path" >&2
    return 1
  }

  case "$path_real" in
    "$parent"/*) ;;
    *)
      echo "error: $label path is outside expected directory: $path" >&2
      return 1
      ;;
  esac
}

require_tracked_file() {
  local file="$1"
  local label="$2"
  local abs dir base rel

  abs="$(resolve_local_path "$file")"
  [[ -e "$abs" ]] || {
    echo "error: benchmark evidence file missing for $label: $file" >&2
    return 1
  }

  dir="$(cd "$(dirname "$abs")" && pwd -P)" || return 1
  base="$(basename "$abs")"
  abs="$dir/$base"

  case "$abs" in
    "$REPO_REAL"/*)
      rel="${abs#$REPO_REAL/}"
      ;;
    *)
      echo "error: benchmark evidence file is outside the repository for $label: $file" >&2
      return 1
      ;;
  esac

  if ! git -C "$REPO_DIR" ls-files --error-unmatch -- "$rel" >/dev/null 2>&1; then
    echo "error: benchmark evidence file is not tracked for $label: $rel" >&2
    return 1
  fi
}

require_repo_commit() {
  local label="$1"
  local commit="$2"

  if [[ ! "$commit" =~ ^[0-9a-f]{40}$ ]]; then
    echo "error: invalid caix commit for $label: $commit" >&2
    return 1
  fi
  if ! git -C "$REPO_DIR" cat-file -e "$commit^{commit}" 2>/dev/null; then
    echo "error: caix commit for $label is not present in this repository: $commit" >&2
    return 1
  fi
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
  local output_label="$4"
  local output

  output="$(resolve_local_path "$output_label")"

  [[ -d "$output" ]] || { echo "error: raw output directory missing for $repo: $output_label" >&2; return 1; }
  [[ -f "$output/metadata.txt" ]] || { echo "error: raw metadata missing for $repo: $output/metadata.txt" >&2; return 1; }
  [[ -f "$output/summary.tsv" ]] || { echo "error: raw summary missing for $repo: $output/summary.tsv" >&2; return 1; }
  require_path_under_dir "$repo raw output" "$output" "$RAW_REAL" || return 1
  local output_real
  output_real="$(canonical_existing_path "$output")" || return 1
  if [[ "$REQUIRE_TRACKED" == "1" ]]; then
    require_tracked_file "$output/metadata.txt" "$repo metadata" || return 1
    require_tracked_file "$output/summary.tsv" "$repo summary" || return 1
  fi

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

  local suite_caix_commit model_caix_commit
  suite_caix_commit="$(metadata_value caix_commit "$suite/metadata.txt")"
  model_caix_commit="$(metadata_value caix_commit "$output/metadata.txt")"
  require_repo_commit "$repo suite metadata" "$suite_caix_commit" || return 1
  require_repo_commit "$repo model metadata" "$model_caix_commit" || return 1
  if [[ "$model_caix_commit" != "$suite_caix_commit" ]]; then
    echo "error: suite/model caix commit drift for $repo: suite=$suite_caix_commit model=$model_caix_commit" >&2
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
  local stdout_path stderr_path stdout_file stderr_file
  while IFS=$'\t' read -r _phase _run _status _generated _load _prefill _decode _tps stdout_path stderr_path; do
    [[ "$_phase" == "measured" && "$_status" == "ok" ]] || continue
    stdout_file="$(resolve_local_path "$stdout_path")"
    stderr_file="$(resolve_local_path "$stderr_path")"
    [[ -f "$stdout_file" ]] || { echo "error: measured stdout missing for $repo: $stdout_path" >&2; return 1; }
    [[ -f "$stderr_file" ]] || { echo "error: measured stderr missing for $repo: $stderr_path" >&2; return 1; }
    require_path_under_dir "$repo measured stdout" "$stdout_file" "$output_real" || return 1
    require_path_under_dir "$repo measured stderr" "$stderr_file" "$output_real" || return 1
    if [[ "$REQUIRE_TRACKED" == "1" ]]; then
      require_tracked_file "$stdout_file" "$repo measured stdout" || return 1
      require_tracked_file "$stderr_file" "$repo measured stderr" || return 1
    fi
    if [[ -z "$first_stdout" ]]; then
      first_stdout="$stdout_file"
    elif ! cmp -s "$first_stdout" "$stdout_file"; then
      echo "error: measured stdout differs for $repo: $first_stdout vs $stdout_path" >&2
      return 1
    fi
  done < "$output/summary.tsv"

  local suite_prompt suite_max_tokens suite_temperature suite_seed suite_raw
  suite_prompt="$(metadata_value prompt "$suite/metadata.txt")"
  suite_max_tokens="$(metadata_value max_tokens "$suite/metadata.txt")"
  suite_temperature="$(metadata_value temperature "$suite/metadata.txt")"
  suite_seed="$(metadata_value seed "$suite/metadata.txt")"
  suite_raw="$(metadata_value raw "$suite/metadata.txt")"

  local model_prompt model_max_tokens model_temperature model_seed model_raw model_mode
  model_prompt="$(metadata_value prompt "$output/metadata.txt")"
  model_max_tokens="$(metadata_value max_tokens "$output/metadata.txt")"
  model_temperature="$(metadata_value temperature "$output/metadata.txt")"
  model_seed="$(metadata_value seed "$output/metadata.txt")"
  model_raw="$(metadata_value raw "$output/metadata.txt")"
  model_mode="$(canonical_benchmark_mode "$(metadata_value benchmark_mode "$output/metadata.txt")")"
  row_mode="$(canonical_benchmark_mode "$row_mode")"

  if [[ "$model_prompt" != "$suite_prompt" || "$model_max_tokens" != "$suite_max_tokens" ||
        "$model_temperature" != "$suite_temperature" || "$model_seed" != "$suite_seed" ||
        "$model_raw" != "$suite_raw" ]]; then
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
  if [[ "$REQUIRE_TRACKED" == "1" ]]; then
    require_tracked_file "$suite/metadata.txt" "$suite metadata"
    require_tracked_file "$suite/summary.tsv" "$suite summary"
  fi
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

  suite_caix_commit="$(metadata_value caix_commit "$suite/metadata.txt")"
  require_repo_commit "$suite suite metadata" "$suite_caix_commit" || exit 1

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
