#!/usr/bin/env bash
# Fail publication checks if local export payloads were left in the checkout.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-export-cleanliness.sh [--exports <dir>] [--report|--removal-plan]

Checks that models/exports contains only .gitkeep and that no export payloads are tracked.
With --report, prints a non-destructive cleanup plan and exits 0.
Does not remove, build, download, upload, or benchmark.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_EXPORTS="$REPO_DIR/models/exports"
EXPORTS="$DEFAULT_EXPORTS"
REPORT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --exports) EXPORTS="${2:?}"; shift 2 ;;
    --report|--removal-plan) REPORT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) echo "unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -d "$EXPORTS" ]] || { echo "error: exports directory not found: $EXPORTS" >&2; exit 2; }
[[ -d "$DEFAULT_EXPORTS" ]] || { echo "error: default exports directory not found: $DEFAULT_EXPORTS" >&2; exit 2; }

EXPORTS_ABS="$(cd "$EXPORTS" && pwd -P)"
DEFAULT_EXPORTS_ABS="$(cd "$DEFAULT_EXPORTS" && pwd -P)"
REPO_ABS="$(cd "$REPO_DIR" && pwd -P)"
case "$EXPORTS_ABS" in
  "$REPO_ABS"/*) ;;
  *)
    echo "error: exports directory must be inside the repository: $EXPORTS_ABS" >&2
    exit 2
    ;;
esac

exports_rel="${EXPORTS_ABS#"$REPO_ABS"/}"

tracked_payloads=()
while IFS= read -r path; do
  tracked_payloads+=("$path")
done < <(
  git -C "$REPO_DIR" ls-files -- "$exports_rel" \
    | awk -v keep="$exports_rel/.gitkeep" '$0 != keep { print }'
)

gitkeep_tracked=1
if ! git -C "$REPO_DIR" ls-files --error-unmatch -- "$exports_rel/.gitkeep" >/dev/null 2>&1; then
  gitkeep_tracked=0
fi

leftovers=()
while IFS= read -r path; do
  leftovers+=("$path")
done < <(find "$EXPORTS_ABS" -mindepth 1 -maxdepth 1 ! -name .gitkeep -print | LC_ALL=C sort)

quote_arg() {
  printf '%q' "$1"
}

print_remove_command() {
  local dry_run="$1"
  local name="$2"

  printf '  scripts/remove-export.sh'
  if [[ "$EXPORTS_ABS" != "$DEFAULT_EXPORTS_ABS" ]]; then
    printf ' --exports %s' "$(quote_arg "$EXPORTS_ABS")"
  fi
  if [[ "$dry_run" == "1" ]]; then
    printf ' --dry-run'
  fi
  printf ' %s\n' "$(quote_arg "$name")"
}

if [[ "$REPORT" == "1" ]]; then
  echo "export cleanliness report"
  echo "exports: $EXPORTS_ABS"

  if [[ "${#tracked_payloads[@]}" -gt 0 ]]; then
    echo
    echo "tracked export payloads:"
    printf '  %s\n' "${tracked_payloads[@]}"
    echo "tracked payloads require git cleanup before publication."
  else
    echo "tracked export payloads: none"
  fi

  if [[ "$gitkeep_tracked" == "1" ]]; then
    echo "$exports_rel/.gitkeep: tracked"
  else
    echo "$exports_rel/.gitkeep: not tracked"
  fi

  if [[ "${#leftovers[@]}" -eq 0 ]]; then
    echo "local export payloads: none"
    echo "report only; no files removed"
    exit 0
  fi

  echo
  echo "local export payloads:"
  printf '  %s\n' "${leftovers[@]}"

  safe_dirs=()
  manual_review=()
  for path in "${leftovers[@]}"; do
    name="$(basename "$path")"
    if [[ -d "$path" && "$name" =~ ^[A-Za-z0-9._-]+$ ]]; then
      safe_dirs+=("$name")
    else
      manual_review+=("$path")
    fi
  done

  if [[ "${#safe_dirs[@]}" -gt 0 ]]; then
    echo
    echo "dry-run cleanup commands:"
    for name in "${safe_dirs[@]}"; do
      print_remove_command 1 "$name"
    done

    echo
    echo "cleanup commands after dry-run review:"
    for name in "${safe_dirs[@]}"; do
      print_remove_command 0 "$name"
    done
  fi

  if [[ "${#manual_review[@]}" -gt 0 ]]; then
    echo
    echo "manual review required; no remove-export command generated for:"
    printf '  %s\n' "${manual_review[@]}"
  fi

  echo "report only; no files removed"
  exit 0
fi

if [[ "${#tracked_payloads[@]}" -gt 0 ]]; then
  echo "error: export payload files are tracked; keep models/exports local-only:" >&2
  printf '%s\n' "${tracked_payloads[@]}" >&2
  exit 1
fi

if [[ "$gitkeep_tracked" != "1" ]]; then
  echo "error: $exports_rel/.gitkeep must be tracked" >&2
  exit 1
fi

if [[ "${#leftovers[@]}" -gt 0 ]]; then
  echo "error: export payloads are still present; remove tested bundles with scripts/remove-export.sh:" >&2
  printf '%s\n' "${leftovers[@]}" >&2
  echo "hint: run scripts/check-export-cleanliness.sh --report for a non-destructive cleanup plan" >&2
  exit 1
fi

echo "export cleanliness ok"
