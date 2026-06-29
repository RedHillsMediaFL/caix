#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/benchmark-model.sh --model <bundle-dir> --name <name> [options]

Options:
  --prompt <text>       Prompt text.
  --prompt-file <path>  Prompt file. Overrides --prompt.
  --max-tokens <n>      Max generated tokens. Default: 128.
  --temperature <n>     Temperature. Default: 0.
  --warmup <n>          Warmup runs. Default: 1.
  --runs <n>            Measured runs. Default: 3.
  --raw                 Skip chat template.
  --draft <dir>         Draft bundle for classic speculative decoding.
  --draft-tokens <n>    Draft tokens proposed per step. Default: 4.
  --repo <id>           Hugging Face repo id for this bundle. Required.
  --repo-revision <sha> Exact model repo commit for this bundle. Required.
  --out <dir>           Output root. Default: benchmarks/raw.
  --force               Ignore an existing stale benchmark lock.

Writes raw stdout/stderr and summary.tsv. Does not publish numbers.
USAGE
}

MODEL=""
NAME=""
PROMPT="Write one factual sentence about local inference on Apple silicon."
PROMPT_FILE=""
MAX_TOKENS=128
TEMPERATURE=0
WARMUP=1
RUNS=3
RAW=0
DRAFT=""
DRAFT_TOKENS=4
REPO=""
REPO_REVISION="unknown"
OUT_ROOT="benchmarks/raw"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:?}"; shift 2 ;;
    --name) NAME="${2:?}"; shift 2 ;;
    --prompt) PROMPT="${2:?}"; shift 2 ;;
    --prompt-file) PROMPT_FILE="${2:?}"; shift 2 ;;
    --max-tokens) MAX_TOKENS="${2:?}"; shift 2 ;;
    --temperature) TEMPERATURE="${2:?}"; shift 2 ;;
    --warmup) WARMUP="${2:?}"; shift 2 ;;
    --runs) RUNS="${2:?}"; shift 2 ;;
    --raw) RAW=1; shift ;;
    --draft) DRAFT="${2:?}"; shift 2 ;;
    --draft-tokens) DRAFT_TOKENS="${2:?}"; shift 2 ;;
    --repo) REPO="${2:?}"; shift 2 ;;
    --repo-revision) REPO_REVISION="${2:?}"; shift 2 ;;
    --out) OUT_ROOT="${2:?}"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$MODEL" ]] || { echo "error: --model is required" >&2; exit 2; }
[[ -n "$NAME" ]] || { echo "error: --name is required" >&2; exit 2; }
[[ -d "$MODEL" ]] || { echo "error: model directory not found: $MODEL" >&2; exit 2; }
if [[ -n "$DRAFT" ]]; then
  [[ -d "$DRAFT" ]] || { echo "error: draft directory not found: $DRAFT" >&2; exit 2; }
fi
[[ "$MAX_TOKENS" =~ ^[0-9]+$ ]] || { echo "error: --max-tokens must be an integer" >&2; exit 2; }
[[ "$WARMUP" =~ ^[0-9]+$ ]] || { echo "error: --warmup must be an integer" >&2; exit 2; }
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "error: --runs must be a positive integer" >&2; exit 2; }
[[ "$DRAFT_TOKENS" =~ ^[1-9][0-9]*$ ]] || { echo "error: --draft-tokens must be a positive integer" >&2; exit 2; }
[[ -n "$REPO" ]] || { echo "error: --repo is required for benchmark evidence" >&2; exit 2; }
[[ "$REPO_REVISION" =~ ^[0-9a-f]{40}$ ]] || {
  echo "error: --repo-revision must be a 40-character commit SHA" >&2
  exit 2
}

if [[ -n "$PROMPT_FILE" ]]; then
  [[ -f "$PROMPT_FILE" ]] || { echo "error: prompt file not found: $PROMPT_FILE" >&2; exit 2; }
  PROMPT="$(<"$PROMPT_FILE")"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib/caix-env.sh"

caix_bin="$(caix_env caix_bin BIN ./caix)"
[[ -x "$caix_bin" ]] || caix_bin="./.build/release/caix"
[[ -x "$caix_bin" ]] || { echo "error: no caix binary found; set caix_bin" >&2; exit 2; }

LOCK="$(caix_env caix_heavy_task_lock HEAVY_TASK_LOCK "$REPO_DIR/.agent-heavy-task.lock")"
DISK_FLOOR_GIB="$(caix_env caix_stop_floor_gib STOP_FLOOR_GIB 500)"
"$SCRIPT_DIR/check-disk-pressure.sh" --path /Volumes/SSD --floor-gib "$DISK_FLOOR_GIB" --quiet

if [[ -e "$LOCK" && "$FORCE" != "1" ]]; then
  echo "error: heavy-task lock exists: $LOCK" >&2
  exit 2
fi
if ps -axo command \
  | grep -E 'coreai\.llm\.export|convert\.py|hf (download|upload)|\.build/(debug|release)/caix (run|eagle)|(^|/)(caix|coreai-pipeline) (run|eagle)|(^|/)swift (build|test)|swift-package|swiftc|swift-frontend|xctest' \
  | grep -v grep >/dev/null; then
  echo "error: another heavy build, conversion, upload, verification, or benchmark is active" >&2
  exit 2
fi

STAMP="$(date '+%Y%m%d-%H%M%S')"
SAFE_NAME="$(printf '%s' "$NAME" | tr -cs 'A-Za-z0-9._-' '-')"
OUT_DIR="$OUT_ROOT/$STAMP-$SAFE_NAME"
mkdir -p "$OUT_DIR"
BENCHMARK_MODE="decode"
[[ -n "$DRAFT" ]] && BENCHMARK_MODE="speculative"

FREE_GIB="$(df -g /Volumes/SSD 2>/dev/null | awk 'NR==2 {print $4}')"
{
  echo "pid=$$"
  echo "task=benchmark $NAME"
  echo "est_peak_disk_gib=1"
  echo "free_gib_at_acquire=${FREE_GIB:-unknown}"
  echo "stop_floor_gib=$DISK_FLOOR_GIB"
  echo "owner=benchmark-model.sh"
  echo "started=$(date '+%Y-%m-%d %H:%M %Z')"
} > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

{
  echo "name=$NAME"
  echo "model=$MODEL"
  echo "draft=$DRAFT"
  echo "repo=$REPO"
  echo "repo_revision=$REPO_REVISION"
  echo "caix_bin=$caix_bin"
  echo "caix_commit=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)"
  echo "git_status=$(git -C "$REPO_DIR" status --short 2>/dev/null | wc -l | tr -d ' ') dirty entries"
  echo "machine=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
  echo "memory_bytes=$(sysctl -n hw.memsize 2>/dev/null || true)"
  echo "os=$(sw_vers -productVersion 2>/dev/null || true) ($(sw_vers -buildVersion 2>/dev/null || true))"
  echo "max_tokens=$MAX_TOKENS"
  echo "temperature=$TEMPERATURE"
  echo "warmup=$WARMUP"
  echo "runs=$RUNS"
  echo "raw=$RAW"
  echo "draft_tokens=$DRAFT_TOKENS"
  echo "benchmark_mode=$BENCHMARK_MODE"
  printf 'prompt=%s\n' "$PROMPT"
} > "$OUT_DIR/metadata.txt"

printf 'phase\trun\tstatus\tgenerated\tload_s\tprefill_s\tdecode_s\tdecode_tps\tstdout\tstderr\n' > "$OUT_DIR/summary.tsv"

run_one() {
  local phase="$1"
  local idx="$2"
  local stdout_file="$OUT_DIR/${phase}-${idx}.stdout.txt"
  local stderr_file="$OUT_DIR/${phase}-${idx}.stderr.txt"
  local args=(run --model "$MODEL" --prompt "$PROMPT" --max-tokens "$MAX_TOKENS" --temperature "$TEMPERATURE" --verbose)
  [[ -n "$DRAFT" ]] && args+=(--draft "$DRAFT" --draft-tokens "$DRAFT_TOKENS")
  [[ "$RAW" == "1" ]] && args+=(--raw)
  local status="ok"
  if ! "$caix_bin" "${args[@]}" >"$stdout_file" 2>"$stderr_file"; then
    status="fail"
  fi
  local summary
  summary="$(grep -E '\[coreai\].*generated.*load=.*prefill=.*decode=.*tok/s' "$stderr_file" | tail -1 || true)"
  local parsed
  parsed="$(printf '%s\n' "$summary" | sed -E 's/.* ([0-9]+) generated,.*load=([0-9.]+)s prefill=([0-9.]+)s decode=([0-9.]+)s \(([0-9.]+) tok\/s\).*/\1\t\2\t\3\t\4\t\5/' || true)"
  if [[ -z "$parsed" || "$parsed" == "$summary" ]]; then
    parsed=$'-\t-\t-\t-\t-'
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$phase" "$idx" "$status" "$parsed" "$stdout_file" "$stderr_file" >> "$OUT_DIR/summary.tsv"
  [[ "$status" == "ok" ]]
}

for ((i = 1; i <= WARMUP; i++)); do
  run_one warmup "$i"
done
for ((i = 1; i <= RUNS; i++)); do
  run_one measured "$i"
done

echo "$OUT_DIR"
