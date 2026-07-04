#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/export-qwen3-samelineage-p0.sh [options]

Exports the same-lineage Qwen3-0.6B noopt/extnone monolithic and staged bundles
needed before the P0 staged parity gate. This is a heavy model conversion path;
run it only in an approved Studio heavy window.

Options:
  --exporter-root <path>            coreai-models exporter repo (default: /Volumes/SSD/ai-dev/coreai-qwen3-stages)
  --python <path>                   exporter Python (default: <exporter-root>/.venv/bin/python)
  --model <id-or-path>              source model (default: Qwen/Qwen3-0.6B)
  --export-dir <path>               output export directory (default: <exporter-root>/exports)
  --monolithic-name <name>          monolithic output name
  --staged-name <name>              staged output name
  --stage-splits <splits>           staged layer splits (default: 0:1,1:2,2:3,3:28)
  --output-dir <path>               evidence directory (default: .tmp/studio-only/<timestamp>-qwen3-samelineage-noopt-export)
  --lock <path>                     heavy lock path (default: CAIX_HEAVY_LOCK or <exporter-root>/.agent-heavy-task.lock)
  --min-free-ram-gb <n>             required free RAM before real export starts (default: 12)
  --min-free-disk-gb <n>            required free disk before real export starts (default: 100)
  --dry-run                         exercise lifecycle/logging without running exports
  -h, --help                        show this help

Environment:
  CAIX_SAMELINEAGE_EXPORT_DRY_RUN=1 is equivalent to --dry-run.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
exporter_root="/Volumes/SSD/ai-dev/coreai-qwen3-stages"
python=""
model="Qwen/Qwen3-0.6B"
export_dir=""
monolithic_name="qwen3-0.6b-coreai-f16-extnone-noopt-samelineage"
staged_name="qwen3-0.6b-coreai-staged-f16-extnone-noopt-first3-samelineage"
stage_splits="0:1,1:2,2:3,3:28"
output_dir="$REPO_DIR/.tmp/studio-only/${timestamp}-qwen3-samelineage-noopt-export"
lock_path=""
min_free_ram_gb="${CAIX_MIN_FREE_RAM_GB:-12}"
min_free_disk_gb="${CAIX_MIN_FREE_DISK_GB:-100}"
dry_run="${CAIX_SAMELINEAGE_EXPORT_DRY_RUN:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exporter-root) exporter_root="${2:?}"; shift 2 ;;
    --python) python="${2:?}"; shift 2 ;;
    --model) model="${2:?}"; shift 2 ;;
    --export-dir) export_dir="${2:?}"; shift 2 ;;
    --monolithic-name) monolithic_name="${2:?}"; shift 2 ;;
    --staged-name) staged_name="${2:?}"; shift 2 ;;
    --stage-splits) stage_splits="${2:?}"; shift 2 ;;
    --output-dir) output_dir="${2:?}"; shift 2 ;;
    --lock) lock_path="${2:?}"; shift 2 ;;
    --min-free-ram-gb) min_free_ram_gb="${2:?}"; shift 2 ;;
    --min-free-disk-gb) min_free_disk_gb="${2:?}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) echo "error: unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

python="${python:-$exporter_root/.venv/bin/python}"
export_dir="${export_dir:-$exporter_root/exports}"
lock_path="${lock_path:-${CAIX_HEAVY_LOCK:-$exporter_root/.agent-heavy-task.lock}}"
log="$output_dir/export.log"
meta="$output_dir/export.meta"

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

print_git_revision() {
  local label="$1"
  local dir="$2"
  local revision
  if revision="$(git -C "$dir" rev-parse HEAD 2>/dev/null)"; then
    printf '%s=%s\n' "$label" "$revision"
  fi
}

print_git_status() {
  local dir="$1"
  if git -C "$dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$dir" status --short
  fi
}

require_positive_int "--min-free-ram-gb" "$min_free_ram_gb"
require_positive_int "--min-free-disk-gb" "$min_free_disk_gb"

if [[ -e "$lock_path" ]]; then
  echo "error: heavy lock exists: $lock_path" >&2
  exit 99
fi
if [[ "$dry_run" != "1" && ! -x "$python" ]]; then
  echo "error: missing exporter python: $python" >&2
  exit 98
fi
if [[ "$dry_run" != "1" && ( -e "$export_dir/$monolithic_name" || -e "$export_dir/$staged_name" ) ]]; then
  echo "error: refusing to overwrite existing same-lineage export outputs" >&2
  echo "  $export_dir/$monolithic_name" >&2
  echo "  $export_dir/$staged_name" >&2
  exit 97
fi
if [[ -e "$output_dir" && -n "$(find "$output_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "error: output directory already exists and is not empty: $output_dir" >&2
  exit 2
fi

observed_free_ram_gb="$(free_ram_gb)"
if [[ "$dry_run" != "1" ]]; then
  if ! awk -v free="$observed_free_ram_gb" -v min="$min_free_ram_gb" 'BEGIN { exit(free >= min ? 0 : 1) }'; then
    echo "error: free RAM ${observed_free_ram_gb}GB is below required ${min_free_ram_gb}GB" >&2
    exit 96
  fi
fi
observed_free_disk_gb="$(free_disk_gb)"
if [[ "$dry_run" != "1" ]]; then
  if ! awk -v free="$observed_free_disk_gb" -v min="$min_free_disk_gb" 'BEGIN { exit(free >= min ? 0 : 1) }'; then
    echo "error: free disk ${observed_free_disk_gb}GB is below required ${min_free_disk_gb}GB" >&2
    exit 95
  fi
fi

mkdir -p "$output_dir" "$(dirname "$lock_path")"

{
  printf 'step=qwen3-samelineage-noopt-export\n'
  printf 'started_utc=%s\n' "$(date -u +%Y%m%dT%H%M%SZ)"
  printf 'repo=%s\n' "$REPO_DIR"
  printf 'exporter_root=%s\n' "$exporter_root"
  print_git_revision caix_commit "$REPO_DIR"
  print_git_revision exporter_commit "$exporter_root"
  if [[ -d "$REPO_DIR/.build/checkouts/coreai-models/.git" ]]; then
    print_git_revision coreai_models_commit "$REPO_DIR/.build/checkouts/coreai-models"
  fi
  printf 'python=%s\n' "$python"
  printf 'model=%s\n' "$model"
  printf 'monolithic_output=%s/%s\n' "$export_dir" "$monolithic_name"
  printf 'staged_output=%s/%s\n' "$export_dir" "$staged_name"
  printf 'stage_splits=%s\n' "$stage_splits"
  printf 'COREAI_MACOS_EXTERNALIZE=none\n'
  printf 'COREAI_MACOS_OPTIMIZE=0\n'
  printf 'COREAI_QWEN3_USE_FUSED_KV=1\n'
  printf 'HF_HOME=/Volumes/SSD/hf-cache\n'
  printf 'offline=1\n'
  printf 'dry_run=%s\n' "$dry_run"
  printf 'min_free_ram_gb=%s\n' "$min_free_ram_gb"
  printf 'observed_free_ram_gb=%s\n' "$observed_free_ram_gb"
  printf 'min_free_disk_gb=%s\n' "$min_free_disk_gb"
  printf 'observed_free_disk_gb=%s\n' "$observed_free_disk_gb"
  printf 'caix_status_start\n'
  print_git_status "$REPO_DIR"
  printf 'caix_status_end\n'
  printf 'exporter_status_start\n'
  print_git_status "$exporter_root"
  printf 'exporter_status_end\n'
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

export COREAI_MACOS_EXTERNALIZE=none
export COREAI_MACOS_OPTIMIZE=0
export COREAI_QWEN3_USE_FUSED_KV=1
export HF_HOME=/Volumes/SSD/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

set +e
(
  date -u
  if [[ "$dry_run" == "1" ]]; then
    printf 'dry_run=1\n'
    printf 'monolithic_exit=0\n'
    printf 'staged_exit=0\n'
    exit 0
  fi

  "$python" -m coreai_models.llm.export "$model" \
    --platform macOS \
    --compression none \
    --compute-precision float16 \
    --output-dir "$export_dir" \
    --output-name "$monolithic_name"
  mono_exit=$?
  printf 'monolithic_exit=%s\n' "$mono_exit"
  if [[ "$mono_exit" -ne 0 ]]; then
    exit "$mono_exit"
  fi

  "$python" -m coreai_models.llm.export "$model" \
    --platform macOS \
    --compression none \
    --compute-precision float16 \
    --output-dir "$export_dir" \
    --output-name "$staged_name" \
    --stage-splits "$stage_splits"
  staged_exit=$?
  printf 'staged_exit=%s\n' "$staged_exit"
  exit "$staged_exit"
) > "$log" 2>&1
exit_code=$?
set -e

finish "$exit_code"

echo "$output_dir"
exit "$exit_code"
