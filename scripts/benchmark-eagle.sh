#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/benchmark-eagle.sh --package <mtp-bundle-dir> --name <name> [options]

Options:
  --prompt <text>       Prompt text.
  --prompt-file <path>  Prompt file. Overrides --prompt.
  --max-tokens <n>      Max generated tokens. Default: 128.
  --warmup <n>          Warmup runs. Default: 1.
  --runs <n>            Measured runs. Default: 3.
  --raw                 Skip chat template.
  --repo <id>           Hugging Face repo id for this package. Required.
  --repo-revision <sha> Exact model repo commit for this package. Required.
  --out <dir>           Output root. Default: benchmarks/raw.
  --draft-tokens <n>    Draft tokens proposed per step. Default: 4.
  --target-only         Run the package target path without draft acceptance.
  --draft-unrolled <dir>
                         Optional unrolled EAGLE draft .aimodel directory.
  --vocab <n>           EAGLE vocabulary size. Default: 262144.
  --backbone <n>        EAGLE hidden size. Default: 2816.
  --sliding-window <n>  EAGLE sliding window. Default: 1024.
  --max-context <n>     EAGLE max context. Default: 4096.
  --force               Ignore an existing stale benchmark lock.

Writes raw stdout/stderr and summary.tsv. Does not publish numbers.
USAGE
}

PACKAGE=""
NAME=""
PROMPT="Write one factual sentence about local inference on Apple silicon."
PROMPT_FILE=""
MAX_TOKENS=128
WARMUP=1
RUNS=3
RAW=0
REPO=""
REPO_REVISION="unknown"
OUT_ROOT="benchmarks/raw"
DRAFT_TOKENS=4
TARGET_ONLY=0
DRAFT_UNROLLED=""
VOCAB=262144
BACKBONE=2816
SLIDING_WINDOW=1024
MAX_CONTEXT=4096
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --package|--model) PACKAGE="${2:?}"; shift 2 ;;
    --name) NAME="${2:?}"; shift 2 ;;
    --prompt) PROMPT="${2:?}"; shift 2 ;;
    --prompt-file) PROMPT_FILE="${2:?}"; shift 2 ;;
    --max-tokens) MAX_TOKENS="${2:?}"; shift 2 ;;
    --warmup) WARMUP="${2:?}"; shift 2 ;;
    --runs) RUNS="${2:?}"; shift 2 ;;
    --raw) RAW=1; shift ;;
    --repo) REPO="${2:?}"; shift 2 ;;
    --repo-revision) REPO_REVISION="${2:?}"; shift 2 ;;
    --out) OUT_ROOT="${2:?}"; shift 2 ;;
    --draft-tokens) DRAFT_TOKENS="${2:?}"; shift 2 ;;
    --target-only) TARGET_ONLY=1; shift ;;
    --draft-unrolled) DRAFT_UNROLLED="${2:?}"; shift 2 ;;
    --vocab|--eagle-vocab) VOCAB="${2:?}"; shift 2 ;;
    --backbone|--hidden-size|--eagle-backbone|--eagle-hidden-size) BACKBONE="${2:?}"; shift 2 ;;
    --sliding-window|--eagle-sliding-window) SLIDING_WINDOW="${2:?}"; shift 2 ;;
    --max-context|--eagle-max-context) MAX_CONTEXT="${2:?}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$PACKAGE" ]] || { echo "error: --package is required" >&2; exit 2; }
[[ -n "$NAME" ]] || { echo "error: --name is required" >&2; exit 2; }
[[ -d "$PACKAGE" ]] || { echo "error: package directory not found: $PACKAGE" >&2; exit 2; }
[[ "$MAX_TOKENS" =~ ^[0-9]+$ ]] || { echo "error: --max-tokens must be an integer" >&2; exit 2; }
[[ "$WARMUP" =~ ^[0-9]+$ ]] || { echo "error: --warmup must be an integer" >&2; exit 2; }
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "error: --runs must be a positive integer" >&2; exit 2; }
[[ "$DRAFT_TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "error: --draft-tokens must be a positive integer" >&2; exit 2; }
[[ "$VOCAB" =~ ^[1-9][0-9]*$ ]] || { echo "error: --vocab must be a positive integer" >&2; exit 2; }
[[ "$BACKBONE" =~ ^[1-9][0-9]*$ ]] || { echo "error: --backbone must be a positive integer" >&2; exit 2; }
[[ "$SLIDING_WINDOW" =~ ^[1-9][0-9]*$ ]] || { echo "error: --sliding-window must be a positive integer" >&2; exit 2; }
[[ "$MAX_CONTEXT" =~ ^[1-9][0-9]*$ ]] || { echo "error: --max-context must be a positive integer" >&2; exit 2; }
[[ -n "$REPO" ]] || { echo "error: --repo is required for benchmark evidence" >&2; exit 2; }
[[ "$REPO_REVISION" =~ ^[0-9a-f]{40}$ ]] || {
  echo "error: --repo-revision must be a 40-character commit SHA" >&2
  exit 2
}

TARGET="$PACKAGE/eagle_target.aimodel"
DRAFT="$PACKAGE/eagle_draft.aimodel"
TOKENIZER="$PACKAGE/tokenizer"
[[ -d "$TARGET" ]] || { echo "error: EAGLE target not found: $TARGET" >&2; exit 2; }
[[ -d "$DRAFT" ]] || { echo "error: EAGLE draft not found: $DRAFT" >&2; exit 2; }
[[ -d "$TOKENIZER" ]] || { echo "error: tokenizer directory not found: $TOKENIZER" >&2; exit 2; }

if [[ -z "$DRAFT_UNROLLED" ]]; then
  for candidate in "$PACKAGE/eagle_draft_unrolled_k${DRAFT_TOKENS}.aimodel" "$PACKAGE/eagle_draft_unrolled.aimodel"; do
    if [[ -d "$candidate" ]]; then
      DRAFT_UNROLLED="$candidate"
      break
    fi
  done
elif [[ ! -d "$DRAFT_UNROLLED" ]]; then
  echo "error: draft-unrolled directory not found: $DRAFT_UNROLLED" >&2
  exit 2
fi

if [[ -n "$PROMPT_FILE" ]]; then
  [[ -f "$PROMPT_FILE" ]] || { echo "error: prompt file not found: $PROMPT_FILE" >&2; exit 2; }
  PROMPT="$(<"$PROMPT_FILE")"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib/caix-env.sh"

if [[ -n "$(git -C "$REPO_DIR" status --short 2>/dev/null)" ]]; then
  echo "error: git worktree must be clean before recording benchmark evidence" >&2
  exit 2
fi

caix_bin="$(caix_env caix_bin BIN ./caix)"
[[ -x "$caix_bin" ]] || caix_bin="./.build/release/caix"
[[ -x "$caix_bin" ]] || { echo "error: no caix binary found; set caix_bin" >&2; exit 2; }

LOCK="$(caix_env caix_heavy_task_lock HEAVY_TASK_LOCK "$REPO_DIR/.agent-heavy-task.lock")"
DISK_FLOOR_GIB="$(caix_env caix_stop_floor_gib STOP_FLOOR_GIB 500)"
"$SCRIPT_DIR/check-disk-pressure.sh" --path /Volumes/SSD --floor-gib "$DISK_FLOOR_GIB" --quiet

guard_args=()
[[ "$FORCE" == "1" ]] && guard_args+=(--ignore-lock)
if [[ "${#guard_args[@]}" -gt 0 ]]; then
  if ! caix_heavy_task_lock="$LOCK" "$SCRIPT_DIR/conversion-guard.sh" "${guard_args[@]}" >/dev/null 2>&1; then
    echo "error: heavy-task guard is busy; another build, conversion, upload, verification, benchmark, or lock is active" >&2
    exit 2
  fi
else
  if ! caix_heavy_task_lock="$LOCK" "$SCRIPT_DIR/conversion-guard.sh" >/dev/null 2>&1; then
    echo "error: heavy-task guard is busy; another build, conversion, upload, verification, benchmark, or lock is active" >&2
    exit 2
  fi
fi

STAMP="$(date '+%Y%m%d-%H%M%S')"
SAFE_NAME="$(printf '%s' "$NAME" | tr -cs 'A-Za-z0-9._-' '-')"
MODE_LABEL="eagle-mtp"
[[ "$TARGET_ONLY" == "1" ]] && MODE_LABEL="eagle-target-only"
OUT_DIR="$OUT_ROOT/$STAMP-$SAFE_NAME-$MODE_LABEL"
mkdir -p "$OUT_DIR"

FREE_GIB="$(df -g /Volumes/SSD 2>/dev/null | awk 'NR==2 {print $4}')"
{
  echo "pid=$$"
  echo "task=benchmark $NAME $MODE_LABEL"
  echo "est_peak_disk_gib=1"
  echo "free_gib_at_acquire=${FREE_GIB:-unknown}"
  echo "stop_floor_gib=$DISK_FLOOR_GIB"
  echo "owner=benchmark-eagle.sh"
  echo "started=$(date '+%Y-%m-%d %H:%M %Z')"
} > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

{
  echo "name=$NAME"
  echo "model=$PACKAGE"
  echo "target=$TARGET"
  echo "draft=$DRAFT"
  echo "draft_unrolled=$DRAFT_UNROLLED"
  echo "tokenizer=$TOKENIZER"
  echo "repo=$REPO"
  echo "repo_revision=$REPO_REVISION"
  echo "caix_bin=$caix_bin"
  echo "caix_commit=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)"
  echo "git_status=$(git -C "$REPO_DIR" status --short 2>/dev/null | wc -l | tr -d ' ') dirty entries"
  echo "machine=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
  echo "memory_bytes=$(sysctl -n hw.memsize 2>/dev/null || true)"
  echo "os=$(sw_vers -productVersion 2>/dev/null || true) ($(sw_vers -buildVersion 2>/dev/null || true))"
  echo "benchmark_mode=$MODE_LABEL"
  echo "max_tokens=$MAX_TOKENS"
  echo "temperature=0"
  echo "warmup=$WARMUP"
  echo "runs=$RUNS"
  echo "raw=$RAW"
  echo "draft_tokens=$DRAFT_TOKENS"
  echo "vocab=$VOCAB"
  echo "backbone=$BACKBONE"
  echo "sliding_window=$SLIDING_WINDOW"
  echo "max_context=$MAX_CONTEXT"
  printf 'prompt=%s\n' "$PROMPT"
} > "$OUT_DIR/metadata.txt"

printf 'phase\trun\tstatus\tgenerated\tload_s\tprefill_s\tdecode_s\tdecode_tps\tstdout\tstderr\n' > "$OUT_DIR/summary.tsv"

run_one() {
  local phase="$1"
  local idx="$2"
  local stdout_file="$OUT_DIR/${phase}-${idx}.stdout.txt"
  local stderr_file="$OUT_DIR/${phase}-${idx}.stderr.txt"
  local args=(eagle
    --target "$TARGET"
    --draft "$DRAFT"
    --tokenizer "$TOKENIZER"
    --prompt "$PROMPT"
    --max-tokens "$MAX_TOKENS"
    --draft-tokens "$DRAFT_TOKENS"
    --vocab "$VOCAB"
    --backbone "$BACKBONE"
    --sliding-window "$SLIDING_WINDOW"
    --max-context "$MAX_CONTEXT")
  [[ "$RAW" == "1" ]] && args+=(--raw)
  [[ "$TARGET_ONLY" == "1" ]] && args+=(--target-only)
  [[ -n "$DRAFT_UNROLLED" ]] && args+=(--draft-unrolled "$DRAFT_UNROLLED")

  local status="ok"
  if ! "$caix_bin" "${args[@]}" >"$stdout_file" 2>"$stderr_file"; then
    status="fail"
  fi
  local summary
  summary="$(grep -E '\[coreai\] eagle: .*generated.*load=.*prefill=.*decode=.*tok/s' "$stderr_file" | tail -1 || true)"
  local parsed
  parsed="$(printf '%s\n' "$summary" | sed -E 's/.* [0-9]+ prompt, ([0-9]+) generated,.*load=([0-9.]+)s prefill=([0-9.]+)s decode=([0-9.]+)s \(([0-9.]+) tok\/s\).*/\1\t\2\t\3\t\4\t\5/' || true)"
  if [[ -z "$parsed" || "$parsed" == "$summary" ]]; then
    parsed=$'-\t-\t-\t-\t-'
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$phase" "$idx" "$status" "$parsed" "$stdout_file" "$stderr_file" >> "$OUT_DIR/summary.tsv"
  [[ "$status" == "ok" ]]
}

verify_measured_runs() {
  local measured failed generated_kinds first_stdout="" stdout_path
  measured="$(awk -F '\t' 'NR > 1 && $1 == "measured" && $3 == "ok" { n++ } END { print n + 0 }' "$OUT_DIR/summary.tsv")"
  failed="$(awk -F '\t' 'NR > 1 && $1 == "measured" && $3 != "ok" { n++ } END { print n + 0 }' "$OUT_DIR/summary.tsv")"
  if [[ "$failed" != "0" ]]; then
    echo "error: measured failures for $REPO: $failed" >&2
    return 1
  fi
  if [[ "$measured" != "$RUNS" ]]; then
    echo "error: measured run count mismatch for $REPO: summary=$measured metadata=$RUNS" >&2
    return 1
  fi
  if awk -F '\t' 'NR > 1 && $1 == "measured" && $3 == "ok" && ($4 !~ /^[0-9]+$/ || $5 !~ /^[0-9]+([.][0-9]+)?$/ || $6 !~ /^[0-9]+([.][0-9]+)?$/ || $7 !~ /^[0-9]+([.][0-9]+)?$/ || $8 !~ /^[0-9]+([.][0-9]+)?$/) { bad = 1 } END { exit bad ? 0 : 1 }' "$OUT_DIR/summary.tsv"; then
    echo "error: measured summary metrics are incomplete for $REPO" >&2
    return 1
  fi
  generated_kinds="$(awk -F '\t' 'NR > 1 && $1 == "measured" && $3 == "ok" { seen[$4] = 1 } END { for (value in seen) n++; print n + 0 }' "$OUT_DIR/summary.tsv")"
  if [[ "$generated_kinds" != "1" ]]; then
    echo "error: measured generated-token count is not stable for $REPO" >&2
    return 1
  fi
  while IFS=$'\t' read -r _phase _run _status _generated _load _prefill _decode _tps stdout_path _stderr; do
    [[ "$_phase" == "measured" && "$_status" == "ok" ]] || continue
    [[ -f "$stdout_path" ]] || { echo "error: measured stdout missing for $REPO: $stdout_path" >&2; return 1; }
    if [[ -z "$first_stdout" ]]; then
      first_stdout="$stdout_path"
    elif ! cmp -s "$first_stdout" "$stdout_path"; then
      echo "error: measured stdout differs for $REPO: $first_stdout vs $stdout_path" >&2
      return 1
    fi
  done < "$OUT_DIR/summary.tsv"
}

for ((i = 1; i <= WARMUP; i++)); do
  run_one warmup "$i"
done
for ((i = 1; i <= RUNS; i++)); do
  run_one measured "$i"
done

verify_measured_runs
echo "$OUT_DIR"
