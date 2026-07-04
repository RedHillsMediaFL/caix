#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run-staged-parity-p0.sh --manifest <stage-manifest.json> [--baseline <bundle> | --expected-sequences <path>] [options]

Runs the heavy P0 same-machine staged parity gate:
  - With --baseline: proves the monolithic baseline is deterministic across fresh Swift test
    processes, then runs staged-vs-monolithic greedy token parity only if the oracle is stable.
  - With --expected-sequences: treats the supplied token sequences as the HF/PyTorch oracle and
    runs teacher-forced staged-vs-expected greedy token parity. Decode feeds the HF token each step
    so legitimate tie flips do not cascade; residual flips pass only when the staged expected-token
    logit is within the configured tie tolerance.

This loads Core AI models. Run only in an approved heavy window.

Options:
  --prompts <path>                  Prompt file (default: docs/distributed-evidence/qwen3-0.6b-prompts.txt)
  --expected-sequences <path>       Expected greedy token sequences file (one prompt per line)
  --tie-tolerance <float>           Max staged top-vs-expected logit gap for a tie (default: 0.02)
  --tie-top-k <n>                   Captured staged top-k logits for tie checks (default: 8)
  --max-tokens <n>                  Greedy tokens per prompt (default: 128)
  --determinism-repeats <n>         Repeats inside each Swift process (default: 3)
  --fresh-process-repeats <n>       Fresh Swift test processes to compare (default: 3)
  --output-dir <dir>                Evidence directory (default: .tmp/staged-parity-p0/<timestamp>)
  --lock <path>                     Heavy lock path (default: CAIX_HEAVY_LOCK or Studio exporter lock)
  --min-free-ram-gb <n>             Required free RAM before starting (default: 12)
  --min-free-disk-gb <n>            Required free disk before starting (default: 100)
  -h, --help                        Show this help.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

baseline=""
manifest=""
prompts="$REPO_DIR/docs/distributed-evidence/qwen3-0.6b-prompts.txt"
expected_sequences=""
tie_tolerance=0.02
tie_top_k=8
max_tokens=128
determinism_repeats=3
fresh_process_repeats=3
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output_dir="$REPO_DIR/.tmp/staged-parity-p0/$timestamp"
lock_path="${CAIX_HEAVY_LOCK:-/Volumes/SSD/ai-dev/coreai-qwen3-stages/.agent-heavy-task.lock}"
min_free_ram_gb=12
min_free_disk_gb=100

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline) baseline="${2:?}"; shift 2 ;;
    --manifest) manifest="${2:?}"; shift 2 ;;
    --prompts) prompts="${2:?}"; shift 2 ;;
    --expected-sequences) expected_sequences="${2:?}"; shift 2 ;;
    --tie-tolerance) tie_tolerance="${2:?}"; shift 2 ;;
    --tie-top-k) tie_top_k="${2:?}"; shift 2 ;;
    --max-tokens) max_tokens="${2:?}"; shift 2 ;;
    --determinism-repeats) determinism_repeats="${2:?}"; shift 2 ;;
    --fresh-process-repeats) fresh_process_repeats="${2:?}"; shift 2 ;;
    --output-dir) output_dir="${2:?}"; shift 2 ;;
    --lock) lock_path="${2:?}"; shift 2 ;;
    --min-free-ram-gb) min_free_ram_gb="${2:?}"; shift 2 ;;
    --min-free-disk-gb) min_free_disk_gb="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) echo "error: unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "error: $name must be a positive integer: $value" >&2
    exit 2
  fi
}

free_ram_gb() {
  vm_stat | awk '
    /page size of/ { page = $8 }
    /Pages free:/ { free = $3; gsub("\\.", "", free) }
    END {
      if (page > 0 && free > 0) {
        printf "%.2f", free * page / 1024 / 1024 / 1024
      } else {
        printf "0"
      }
    }'
}

free_disk_gb() {
  df -k /Volumes/SSD | awk 'NR == 2 { printf "%.2f", $4 / 1024 / 1024 }'
}

require_positive_int "--max-tokens" "$max_tokens"
require_positive_int "--determinism-repeats" "$determinism_repeats"
require_positive_int "--fresh-process-repeats" "$fresh_process_repeats"
require_positive_int "--min-free-ram-gb" "$min_free_ram_gb"
require_positive_int "--min-free-disk-gb" "$min_free_disk_gb"
require_positive_int "--tie-top-k" "$tie_top_k"

if [[ -z "$manifest" ]]; then
  echo "error: --manifest is required" >&2
  usage >&2
  exit 2
fi
oracle_modes=0
[[ -n "$baseline" ]] && oracle_modes=$((oracle_modes + 1))
[[ -n "$expected_sequences" ]] && oracle_modes=$((oracle_modes + 1))
if [[ "$oracle_modes" -eq 0 ]]; then
  echo "error: one oracle mode is required: --baseline or --expected-sequences" >&2
  usage >&2
  exit 2
fi
if [[ "$oracle_modes" -gt 1 ]]; then
  echo "error: use only one oracle mode: --baseline or --expected-sequences" >&2
  usage >&2
  exit 2
fi
if [[ -n "$baseline" && ! -d "$baseline" ]]; then
  echo "error: missing baseline bundle: $baseline" >&2
  exit 2
fi
if [[ -n "$expected_sequences" && ! -f "$expected_sequences" ]]; then
  echo "error: missing expected token sequences file: $expected_sequences" >&2
  exit 2
fi
if [[ ! -f "$manifest" ]]; then
  echo "error: missing staged manifest: $manifest" >&2
  exit 2
fi
if [[ ! -f "$prompts" ]]; then
  echo "error: missing prompt file: $prompts" >&2
  exit 2
fi
if [[ -e "$output_dir" && -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "error: output directory already exists and is not empty: $output_dir" >&2
  exit 2
fi
if [[ -e "$lock_path" ]]; then
  echo "error: heavy lock exists: $lock_path" >&2
  exit 99
fi

observed_free_ram_gb="$(free_ram_gb)"
if ! awk -v free="$observed_free_ram_gb" -v min="$min_free_ram_gb" 'BEGIN { exit(free >= min ? 0 : 1) }'; then
  echo "error: free RAM ${observed_free_ram_gb}GB is below required ${min_free_ram_gb}GB" >&2
  exit 96
fi
observed_free_disk_gb="$(free_disk_gb)"
if ! awk -v free="$observed_free_disk_gb" -v min="$min_free_disk_gb" 'BEGIN { exit(free >= min ? 0 : 1) }'; then
  echo "error: free disk ${observed_free_disk_gb}GB is below required ${min_free_disk_gb}GB" >&2
  exit 95
fi

mkdir -p "$output_dir" "$(dirname "$lock_path")"
meta="$output_dir/p0-gates.meta"

{
  printf 'step=staged-parity-p0\n'
  printf 'started_utc=%s\n' "$(date -u +%Y%m%dT%H%M%SZ)"
  printf 'repo=%s\n' "$REPO_DIR"
  printf 'caix_commit=%s\n' "$(git -C "$REPO_DIR" rev-parse HEAD)"
  if [[ -d "$REPO_DIR/.build/checkouts/coreai-models/.git" ]]; then
    printf 'coreai_models_commit=%s\n' "$(git -C "$REPO_DIR/.build/checkouts/coreai-models" rev-parse HEAD)"
  fi
  printf 'baseline=%s\n' "$baseline"
  printf 'expected_sequences=%s\n' "$expected_sequences"
  printf 'tie_tolerance=%s\n' "$tie_tolerance"
  printf 'tie_top_k=%s\n' "$tie_top_k"
  printf 'manifest=%s\n' "$manifest"
  printf 'prompts=%s\n' "$prompts"
  printf 'max_tokens=%s\n' "$max_tokens"
  printf 'determinism_repeats_per_process=%s\n' "$determinism_repeats"
  printf 'fresh_process_repeats=%s\n' "$fresh_process_repeats"
  printf 'min_free_ram_gb=%s\n' "$min_free_ram_gb"
  printf 'observed_free_ram_gb=%s\n' "$observed_free_ram_gb"
  printf 'min_free_disk_gb=%s\n' "$min_free_disk_gb"
  printf 'observed_free_disk_gb=%s\n' "$observed_free_disk_gb"
  printf 'git_status_start\n'
  git -C "$REPO_DIR" status --short
  printf 'git_status_end\n'
  df -h /Volumes/SSD
  vm_stat
} > "$meta"

printf 'pid=%s\nout=%s\n' "$$" "$output_dir" > "$lock_path"
finalized=0
finish() {
  local status="$1"
  if [[ "$finalized" -eq 1 ]]; then
    return
  fi
  finalized=1
  {
    printf 'exit_status=%s\n' "$status"
    printf 'ended_utc=%s\n' "$(date -u +%Y%m%dT%H%M%SZ)"
    df -h /Volumes/SSD
    vm_stat
  } >> "$meta"
}
cleanup() {
  local status=$?
  finish "$status"
  rm -f "$lock_path"
}
trap cleanup EXIT

run_swift_gate() {
  local log="$1"
  local filter="$2"
  local observed_tokens="${3:-}"
  (
    cd "$REPO_DIR"
    export COREAI_RUNTIME=1
    export CAIX_STAGE_MANIFEST="$manifest"
    export CAIX_TOKEN_MATCH_PROMPTS="$prompts"
    export CAIX_TOKEN_MATCH_MAX_TOKENS="$max_tokens"
    export CAIX_MONOLITHIC_DETERMINISM_REPEATS="$determinism_repeats"
    if [[ -n "$baseline" ]]; then
      export CAIX_BASELINE_MODEL="$baseline"
    fi
    if [[ -n "$observed_tokens" ]]; then
      export CAIX_OBSERVED_GREEDY_TOKEN_SEQUENCES_FILE="$observed_tokens"
    fi
    nice -n 10 swift test -c release --filter "$filter"
  ) > "$log" 2>&1
}

run_expected_gate() {
  local log="$1"
  (
    cd "$REPO_DIR"
    export COREAI_RUNTIME=1
    export CAIX_STAGE_MANIFEST="$manifest"
    export CAIX_TOKEN_MATCH_PROMPTS="$prompts"
    export CAIX_TOKEN_MATCH_MAX_TOKENS="$max_tokens"
    export CAIX_EXPECTED_GREEDY_TOKEN_SEQUENCES_FILE="$expected_sequences"
    export CAIX_EXPECTED_GREEDY_TIE_TOLERANCE="$tie_tolerance"
    export CAIX_CAPTURE_STAGE_TOPK="$tie_top_k"
    nice -n 10 swift test -c release --filter \
      DistributedCoreAIStageAssetIntegrationTests/testRealStageAssetsMatchExpectedGreedyTokenSequences
  ) > "$log" 2>&1
}

if [[ -n "$expected_sequences" ]]; then
  parity_name=staged_expected_128
  printf '%s_start_utc=%s\n' "$parity_name" "$(date -u +%Y%m%dT%H%M%SZ)" >> "$meta"
  set +e
  run_expected_gate "$output_dir/${parity_name}.log"
  parity_exit=$?
  set -e
  printf '%s_exit=%s\n' "$parity_name" "$parity_exit" >> "$meta"
  printf '%s_end_utc=%s\n' "$parity_name" "$(date -u +%Y%m%dT%H%M%SZ)" >> "$meta"
  finish "$parity_exit"
  echo "$output_dir"
  exit "$parity_exit"
fi

determinism_name=monolithic_determinism
printf '%s_start_utc=%s\n' "$determinism_name" "$(date -u +%Y%m%dT%H%M%SZ)" >> "$meta"
reference_tokens=""
for index in $(seq 1 "$fresh_process_repeats"); do
  log="$output_dir/${determinism_name}_${index}.log"
  tokens="$output_dir/${determinism_name}_${index}.tokens"
  set +e
  run_swift_gate "$log" \
    DistributedCoreAIStageAssetIntegrationTests/testRealMonolithicGreedyTokenSequencesAreDeterministic \
    "$tokens"
  exit_code=$?
  set -e
  printf '%s_process_%s_exit=%s\n' "$determinism_name" "$index" "$exit_code" >> "$meta"
  if [[ "$exit_code" -ne 0 ]]; then
    printf '%s_exit=%s\n' "$determinism_name" "$exit_code" >> "$meta"
    printf '%s_end_utc=%s\n' "$determinism_name" "$(date -u +%Y%m%dT%H%M%SZ)" >> "$meta"
    exit "$exit_code"
  fi
  if [[ ! -s "$tokens" ]]; then
    printf '%s_exit=3\n' "$determinism_name" >> "$meta"
    printf '%s_missing_tokens=%s\n' "$determinism_name" "$tokens" >> "$meta"
    printf '%s_end_utc=%s\n' "$determinism_name" "$(date -u +%Y%m%dT%H%M%SZ)" >> "$meta"
    exit 3
  fi
  if [[ -z "$reference_tokens" ]]; then
    reference_tokens="$tokens"
    continue
  fi
  if ! cmp -s "$reference_tokens" "$tokens"; then
    diff -u "$reference_tokens" "$tokens" > "$output_dir/${determinism_name}_cross_process.diff" || true
    printf '%s_exit=4\n' "$determinism_name" >> "$meta"
    printf '%s_mismatch_reference=%s\n' "$determinism_name" "$reference_tokens" >> "$meta"
    printf '%s_mismatch_observed=%s\n' "$determinism_name" "$tokens" >> "$meta"
    printf '%s_end_utc=%s\n' "$determinism_name" "$(date -u +%Y%m%dT%H%M%SZ)" >> "$meta"
    exit 4
  fi
done
printf '%s_exit=0\n' "$determinism_name" >> "$meta"
printf '%s_reference_tokens=%s\n' "$determinism_name" "$reference_tokens" >> "$meta"
printf '%s_end_utc=%s\n' "$determinism_name" "$(date -u +%Y%m%dT%H%M%SZ)" >> "$meta"

parity_name=staged_monolithic_128
printf '%s_start_utc=%s\n' "$parity_name" "$(date -u +%Y%m%dT%H%M%SZ)" >> "$meta"
set +e
run_swift_gate "$output_dir/${parity_name}.log" \
  DistributedCoreAIStageAssetIntegrationTests/testRealStageAssetsMatchMonolithicGreedyTokens
parity_exit=$?
set -e
printf '%s_exit=%s\n' "$parity_name" "$parity_exit" >> "$meta"
printf '%s_end_utc=%s\n' "$parity_name" "$(date -u +%Y%m%dT%H%M%SZ)" >> "$meta"

finish "$parity_exit"

echo "$output_dir"
exit "$parity_exit"
