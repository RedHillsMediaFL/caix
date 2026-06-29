#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/benchmark-suite.sh [options]

Options:
  --manifest <path>     TSV manifest. Default: benchmarks/MANIFEST.tsv.
  --revisions <path>    Optional TSV: repo<TAB>revision.
  --exports <dir>       Local bundle root. Default: models/exports.
  --out <dir>           Output root. Default: benchmarks/raw.
  --prompt <text>       Prompt text for every run.
  --prompt-file <path>  Prompt file. Overrides --prompt.
  --repo-revision <sha> Exact model repo commit to record for every measured row. Default: unknown.
  --max-tokens <n>      Max generated tokens. Default: 128.
  --temperature <n>     Temperature. Default: 0.
  --warmup <n>          Warmup runs. Default: 1.
  --runs <n>            Measured runs. Default: 3.
  --raw                 Skip chat template for every run.
  --dry-run             Write the suite summary without launching caix.

Every manifest row is recorded as measured, planned, or skipped with a reason.
This script does not download models and does not publish numbers.
Rows with benchmark_mode=speculative use scripts/benchmark-model.sh with <bundle>/draft.
Rows with benchmark_mode=eagle use scripts/benchmark-eagle.sh against package layouts that contain
eagle_target.aimodel, eagle_draft.aimodel, and tokenizer/.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MANIFEST="$REPO_DIR/benchmarks/MANIFEST.tsv"
REVISIONS=""
EXPORTS="$REPO_DIR/models/exports"
OUT_ROOT="$REPO_DIR/benchmarks/raw"
PROMPT="Write one factual sentence about local inference on Apple silicon."
PROMPT_FILE=""
REPO_REVISION="unknown"
MAX_TOKENS=128
TEMPERATURE=0
WARMUP=1
RUNS=3
RAW=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="${2:?}"; shift 2 ;;
    --revisions) REVISIONS="${2:?}"; shift 2 ;;
    --exports) EXPORTS="${2:?}"; shift 2 ;;
    --out) OUT_ROOT="${2:?}"; shift 2 ;;
    --prompt) PROMPT="${2:?}"; shift 2 ;;
    --prompt-file) PROMPT_FILE="${2:?}"; shift 2 ;;
    --repo-revision) REPO_REVISION="${2:?}"; shift 2 ;;
    --max-tokens) MAX_TOKENS="${2:?}"; shift 2 ;;
    --temperature) TEMPERATURE="${2:?}"; shift 2 ;;
    --warmup) WARMUP="${2:?}"; shift 2 ;;
    --runs) RUNS="${2:?}"; shift 2 ;;
    --raw) RAW=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$MANIFEST" ]] || { echo "error: manifest not found: $MANIFEST" >&2; exit 2; }
if [[ -n "$REVISIONS" ]]; then
  [[ -f "$REVISIONS" ]] || { echo "error: revisions file not found: $REVISIONS" >&2; exit 2; }
fi
[[ -d "$EXPORTS" ]] || { echo "error: exports directory not found: $EXPORTS" >&2; exit 2; }
[[ "$MAX_TOKENS" =~ ^[0-9]+$ ]] || { echo "error: --max-tokens must be an integer" >&2; exit 2; }
[[ "$WARMUP" =~ ^[0-9]+$ ]] || { echo "error: --warmup must be an integer" >&2; exit 2; }
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "error: --runs must be a positive integer" >&2; exit 2; }

if [[ -n "$PROMPT_FILE" ]]; then
  [[ -f "$PROMPT_FILE" ]] || { echo "error: prompt file not found: $PROMPT_FILE" >&2; exit 2; }
  PROMPT="$(<"$PROMPT_FILE")"
fi

DISK_FLOOR_GIB="${CAIX_STOP_FLOOR_GIB:-500}"
"$SCRIPT_DIR/check-disk-pressure.sh" --path /Volumes/SSD --floor-gib "$DISK_FLOOR_GIB" --quiet

revision_for_repo() {
  local repo="$1"
  [[ -n "$REVISIONS" ]] || return 0
  awk -F '\t' -v repo="$repo" '
    $1 == repo {
      print $2
      found = 1
      exit
    }
    END {
      if (!found) exit 0
    }
  ' "$REVISIONS"
}

heavy_task_guard() {
  local lock="$REPO_DIR/.agent-heavy-task.lock"
  if [[ -e "$lock" ]]; then
    echo "error: heavy-task lock exists: $lock" >&2
    return 2
  fi
  if ps -axo command \
    | grep -E 'coreai\.llm\.export|convert\.py|hf (download|upload)|\.build/(debug|release)/caix (run|eagle)|(^|/)(caix|coreai-pipeline) (run|eagle)|(^|/)swift (build|test)|swift-package|swiftc|swift-frontend|xctest' \
    | grep -v grep >/dev/null; then
    echo "error: another heavy build, conversion, upload, verification, or benchmark is active" >&2
    return 2
  fi
}

STAMP="$(date '+%Y%m%d-%H%M%S')"
SUITE_DIR="$OUT_ROOT/$STAMP-suite"
mkdir -p "$SUITE_DIR"
SUMMARY="$SUITE_DIR/summary.tsv"

{
  echo "manifest=$MANIFEST"
  echo "revisions=$REVISIONS"
  echo "exports=$EXPORTS"
  echo "caix_commit=$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)"
  echo "git_status=$(git -C "$REPO_DIR" status --short 2>/dev/null | wc -l | tr -d ' ') dirty entries"
  echo "machine=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || true)"
  echo "memory_bytes=$(sysctl -n hw.memsize 2>/dev/null || true)"
  echo "os=$(sw_vers -productVersion 2>/dev/null || true) ($(sw_vers -buildVersion 2>/dev/null || true))"
  echo "max_tokens=$MAX_TOKENS"
  echo "temperature=$TEMPERATURE"
  echo "repo_revision=$REPO_REVISION"
  echo "warmup=$WARMUP"
  echo "runs=$RUNS"
  echo "raw=$RAW"
  echo "dry_run=$DRY_RUN"
  printf 'prompt=%s\n' "$PROMPT"
} > "$SUITE_DIR/metadata.txt"

printf 'repo\tlocal_dir\tkind\tbenchmark_mode\tstatus\treason\tbundle\toutput\n' > "$SUMMARY"

total=0
measured=0
planned=0
skipped=0
failed=0

while IFS=$'\t' read -r repo local_dir kind mode status notes; do
  [[ -z "${repo:-}" || "$repo" == "repo" || "$repo" == \#* ]] && continue
  total=$((total + 1))

  bundle="$EXPORTS/$local_dir"
  output="-"
  reason="-"
  row_status="skip"

  if [[ "$status" != "eligible" ]]; then
    reason="$status"
    [[ -n "${notes:-}" ]] && reason="$reason: $notes"
    skipped=$((skipped + 1))
  elif [[ "$mode" != "decode" && "$mode" != "speculative" && "$mode" != "eagle" ]]; then
    reason="unsupported benchmark mode: $mode"
    skipped=$((skipped + 1))
  elif [[ ! -d "$bundle" ]]; then
    reason="missing local bundle"
    skipped=$((skipped + 1))
  elif [[ "$mode" == "speculative" && ! -d "$bundle/draft" ]]; then
    reason="missing draft bundle"
    skipped=$((skipped + 1))
  elif [[ "$mode" == "eagle" && ! -d "$bundle/eagle_target.aimodel" ]]; then
    reason="missing EAGLE target package"
    skipped=$((skipped + 1))
  elif [[ "$mode" == "eagle" && ! -d "$bundle/eagle_draft.aimodel" ]]; then
    reason="missing EAGLE draft package"
    skipped=$((skipped + 1))
  elif [[ "$mode" == "eagle" && ! -d "$bundle/tokenizer" ]]; then
    reason="missing EAGLE tokenizer"
    skipped=$((skipped + 1))
  elif [[ "$DRY_RUN" == "1" ]]; then
    row_status="planned"
    reason="dry run"
    planned=$((planned + 1))
  else
    heavy_task_guard
    revision="$(revision_for_repo "$repo")"
    row_revision="$REPO_REVISION"
    [[ -n "$revision" ]] && row_revision="$revision"
    if [[ "$mode" == "eagle" ]]; then
      cmd=("$SCRIPT_DIR/benchmark-eagle.sh"
        --package "$bundle"
        --name "$local_dir"
        --repo "$repo"
        --repo-revision "$row_revision"
        --prompt "$PROMPT"
        --max-tokens "$MAX_TOKENS"
        --warmup "$WARMUP"
        --runs "$RUNS"
        --out "$OUT_ROOT")
    else
      cmd=("$SCRIPT_DIR/benchmark-model.sh"
        --model "$bundle"
        --name "$local_dir"
        --repo "$repo"
        --repo-revision "$row_revision"
        --prompt "$PROMPT"
        --max-tokens "$MAX_TOKENS"
        --temperature "$TEMPERATURE"
        --warmup "$WARMUP"
        --runs "$RUNS"
        --out "$OUT_ROOT")
      [[ "$mode" == "speculative" ]] && cmd+=(--draft "$bundle/draft" --draft-tokens 4)
    fi
    [[ "$RAW" == "1" ]] && cmd+=(--raw)

    if output="$("${cmd[@]}")"; then
      row_status="measured"
      measured=$((measured + 1))
    else
      row_status="fail"
      reason="benchmark command failed"
      failed=$((failed + 1))
    fi
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$repo" "$local_dir" "$kind" "$mode" "$row_status" "$reason" "$bundle" "$output" >> "$SUMMARY"
done < "$MANIFEST"

{
  echo "total=$total"
  echo "measured=$measured"
  echo "planned=$planned"
  echo "skipped=$skipped"
  echo "failed=$failed"
} >> "$SUITE_DIR/metadata.txt"

echo "$SUITE_DIR"
