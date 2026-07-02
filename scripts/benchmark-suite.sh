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
  --seed <n>            RNG seed passed to caix run rows.
  --warmup <n>          Warmup runs. Default: 1.
  --runs <n>            Measured runs. Default: 3.
  --raw                 Skip chat template for every run.
  --dry-run             Write the suite summary without launching caix.
  --force               Hand an existing benchmark lock to the child runner.

Every manifest row is recorded as measured, planned, or skipped with a reason.
This script does not download models and does not publish numbers.
Rows with benchmark_mode=speculative use scripts/benchmark-model.sh with <bundle>/draft.
Rows with benchmark_mode=eagle-mtp use scripts/benchmark-eagle.sh against package layouts that contain
eagle_target.aimodel, eagle_draft.aimodel, and tokenizer/.
Non-dry-run measured rows require a 40-character model repo revision from --revisions or
--repo-revision.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib/caix-env.sh"

LOCK="$(caix_env caix_heavy_task_lock HEAVY_TASK_LOCK "$REPO_DIR/.agent-heavy-task.lock")"

MANIFEST="$REPO_DIR/benchmarks/MANIFEST.tsv"
REVISIONS=""
EXPORTS="$REPO_DIR/models/exports"
OUT_ROOT="$REPO_DIR/benchmarks/raw"
PROMPT="Write one factual sentence about local inference on Apple silicon."
PROMPT_FILE=""
REPO_REVISION="unknown"
MAX_TOKENS=128
TEMPERATURE=0
SEED=""
WARMUP=1
RUNS=3
RAW=0
DRY_RUN=0
FORCE=0

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
    --seed) SEED="${2:?}"; shift 2 ;;
    --warmup) WARMUP="${2:?}"; shift 2 ;;
    --runs) RUNS="${2:?}"; shift 2 ;;
    --raw) RAW=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
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
if [[ -n "$SEED" && ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "error: --seed must be a non-negative integer" >&2
  exit 2
fi
[[ "$WARMUP" =~ ^[0-9]+$ ]] || { echo "error: --warmup must be an integer" >&2; exit 2; }
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "error: --runs must be a positive integer" >&2; exit 2; }
if [[ "$REPO_REVISION" != "unknown" && ! "$REPO_REVISION" =~ ^[0-9a-f]{40}$ ]]; then
  echo "error: --repo-revision must be a 40-character commit SHA or omitted" >&2
  exit 2
fi

if [[ -n "$PROMPT_FILE" ]]; then
  [[ -f "$PROMPT_FILE" ]] || { echo "error: prompt file not found: $PROMPT_FILE" >&2; exit 2; }
  PROMPT="$(<"$PROMPT_FILE")"
fi

DISK_FLOOR_GIB="$(caix_env caix_stop_floor_gib STOP_FLOOR_GIB 500)"
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

canonical_benchmark_mode() {
  case "$1" in
    eagle) printf 'eagle-mtp' ;;
    *) printf '%s' "$1" ;;
  esac
}

require_revision_for_measured_rows() {
  local errors=0
  local repo local_dir kind mode status notes canonical_mode bundle revision row_revision
  while IFS=$'\t' read -r repo local_dir kind mode status notes; do
    [[ -z "${repo:-}" || "$repo" == "repo" || "$repo" == \#* ]] && continue
    [[ "$status" == "eligible" ]] || continue
    canonical_mode="$(canonical_benchmark_mode "$mode")"
    [[ "$canonical_mode" == "decode" || "$canonical_mode" == "speculative" || "$canonical_mode" == "eagle-mtp" ]] || continue

    bundle="$EXPORTS/$local_dir"
    [[ -d "$bundle" ]] || continue
    [[ "$canonical_mode" != "speculative" || -d "$bundle/draft" ]] || continue
    if [[ "$canonical_mode" == "eagle-mtp" ]]; then
      [[ -d "$bundle/eagle_target.aimodel" && -d "$bundle/eagle_draft.aimodel" && -d "$bundle/tokenizer" ]] || continue
    fi

    revision="$(revision_for_repo "$repo")"
    row_revision="$REPO_REVISION"
    [[ -n "$revision" ]] && row_revision="$revision"
    if [[ ! "$row_revision" =~ ^[0-9a-f]{40}$ ]]; then
      echo "error: measured benchmark row needs exact repo revision: $repo" >&2
      errors=$((errors + 1))
    fi
  done < "$MANIFEST"

  [[ "$errors" == "0" ]] || return 1
}

require_seed_supported_for_measured_rows() {
  [[ -n "$SEED" ]] || return 0

  local errors=0
  local repo local_dir kind mode status notes canonical_mode bundle
  while IFS=$'\t' read -r repo local_dir kind mode status notes; do
    [[ -z "${repo:-}" || "$repo" == "repo" || "$repo" == \#* ]] && continue
    [[ "$status" == "eligible" ]] || continue
    canonical_mode="$(canonical_benchmark_mode "$mode")"
    [[ "$canonical_mode" == "eagle-mtp" ]] || continue

    bundle="$EXPORTS/$local_dir"
    [[ -d "$bundle/eagle_target.aimodel" && -d "$bundle/eagle_draft.aimodel" && -d "$bundle/tokenizer" ]] || continue

    echo "error: --seed is not supported for EAGLE benchmark rows: $repo" >&2
    errors=$((errors + 1))
  done < "$MANIFEST"

  [[ "$errors" == "0" ]] || return 1
}

eagle_backbone_for_bundle() {
  local bundle="$1"
  local contract="$bundle/contract.txt"
  local backbone=""
  if [[ -f "$contract" ]]; then
    backbone="$(sed -nE 's/.*hidden\[f16,1xNx([0-9]+)\].*/\1/p' "$contract" | head -1)"
  fi
  if [[ "$backbone" =~ ^[1-9][0-9]*$ ]]; then
    printf '%s' "$backbone"
  else
    printf '2816'
  fi
}

heavy_task_guard() {
  local guard_args=()
  [[ "$FORCE" == "1" ]] && guard_args+=(--ignore-lock)
  if [[ "${#guard_args[@]}" -gt 0 ]]; then
    if ! caix_heavy_task_lock="$LOCK" "$SCRIPT_DIR/conversion-guard.sh" "${guard_args[@]}" >/dev/null 2>&1; then
      echo "error: heavy-task guard is busy; another build, conversion, upload, verification, benchmark, or lock is active" >&2
      return 2
    fi
  else
    if ! caix_heavy_task_lock="$LOCK" "$SCRIPT_DIR/conversion-guard.sh" >/dev/null 2>&1; then
      echo "error: heavy-task guard is busy; another build, conversion, upload, verification, benchmark, or lock is active" >&2
      return 2
    fi
  fi
}

if [[ "$DRY_RUN" != "1" ]]; then
  if [[ -n "$(git -C "$REPO_DIR" status --short 2>/dev/null)" ]]; then
    echo "error: git worktree must be clean before recording benchmark evidence" >&2
    exit 2
  fi
  require_revision_for_measured_rows
  require_seed_supported_for_measured_rows
  heavy_task_guard
fi

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
  echo "seed=$SEED"
  echo "repo_revision=$REPO_REVISION"
  echo "warmup=$WARMUP"
  echo "runs=$RUNS"
  echo "raw=$RAW"
  echo "dry_run=$DRY_RUN"
  echo "force=$FORCE"
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
  canonical_mode="$(canonical_benchmark_mode "$mode")"
  is_eagle=0
  [[ "$canonical_mode" == "eagle-mtp" ]] && is_eagle=1

  bundle="$EXPORTS/$local_dir"
  output="-"
  reason="-"
  row_status="skip"

  if [[ "$status" != "eligible" ]]; then
    reason="$status"
    [[ -n "${notes:-}" ]] && reason="$reason: $notes"
    skipped=$((skipped + 1))
  elif [[ "$canonical_mode" != "decode" && "$canonical_mode" != "speculative" && "$canonical_mode" != "eagle-mtp" ]]; then
    reason="unsupported benchmark mode: $mode"
    skipped=$((skipped + 1))
  elif [[ ! -d "$bundle" ]]; then
    reason="missing local bundle"
    skipped=$((skipped + 1))
  elif [[ "$canonical_mode" == "speculative" && ! -d "$bundle/draft" ]]; then
    reason="missing draft bundle"
    skipped=$((skipped + 1))
  elif [[ "$is_eagle" == "1" && ! -d "$bundle/eagle_target.aimodel" ]]; then
    reason="missing EAGLE target package"
    skipped=$((skipped + 1))
  elif [[ "$is_eagle" == "1" && ! -d "$bundle/eagle_draft.aimodel" ]]; then
    reason="missing EAGLE draft package"
    skipped=$((skipped + 1))
  elif [[ "$is_eagle" == "1" && ! -d "$bundle/tokenizer" ]]; then
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
    if [[ "$is_eagle" == "1" ]]; then
      eagle_backbone="$(eagle_backbone_for_bundle "$bundle")"
      cmd=("$SCRIPT_DIR/benchmark-eagle.sh"
        --package "$bundle"
        --name "$local_dir"
        --repo "$repo"
        --repo-revision "$row_revision"
        --prompt "$PROMPT"
        --max-tokens "$MAX_TOKENS"
        --backbone "$eagle_backbone"
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
      [[ -n "$SEED" ]] && cmd+=(--seed "$SEED")
      [[ "$canonical_mode" == "speculative" ]] && cmd+=(--draft "$bundle/draft" --draft-tokens 4)
    fi
    [[ "$RAW" == "1" ]] && cmd+=(--raw)
    [[ "$FORCE" == "1" ]] && cmd+=(--force)

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
    "$repo" "$local_dir" "$kind" "$canonical_mode" "$row_status" "$reason" "$bundle" "$output" >> "$SUMMARY"
done < "$MANIFEST"

{
  echo "total=$total"
  echo "measured=$measured"
  echo "planned=$planned"
  echo "skipped=$skipped"
  echo "failed=$failed"
} >> "$SUITE_DIR/metadata.txt"

echo "$SUITE_DIR"
