#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-publication-gates.sh [--hub] [--distributed] [--caix <path>] [--brew-caix <path>]

Runs the non-heavy gates for publishing docs, cards, manifests, and benchmark evidence.
Default checks are local only. --hub adds Hugging Face metadata/model-card checks.
--distributed checks the Thunderbolt readiness gate.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_HUB=0
RUN_DISTRIBUTED=0
caix_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hub) RUN_HUB=1; shift ;;
    --distributed) RUN_DISTRIBUTED=1; shift ;;
    --caix) caix_args+=(--caix "${2:?}"); shift 2 ;;
    --brew-caix) caix_args+=(--brew-caix "${2:?}"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) echo "error: unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

run() {
  printf '==> %s\n' "$*"
  "$@"
}

run git -C "$REPO_DIR" diff --check HEAD --
shell_scripts=()
while IFS= read -r script; do
  shell_scripts+=("$script")
done < <(find "$SCRIPT_DIR" -maxdepth 1 -type f -name '*.sh' | sort)
if [[ "${#shell_scripts[@]}" -gt 0 ]]; then
  run bash -n "${shell_scripts[@]}"
fi
run "$SCRIPT_DIR/check-token-handling.sh"
run "$SCRIPT_DIR/check-public-copy.sh"
run "$SCRIPT_DIR/check-version-sync.sh"
run "$SCRIPT_DIR/check-conversion-ledger.sh"
run "$SCRIPT_DIR/check-conversion-gap-audit.sh"
run "$SCRIPT_DIR/check-benchmark-raw.sh"
run "$SCRIPT_DIR/check-tester-requests.sh"

if [[ "$RUN_HUB" == "1" ]]; then
  export HF_HOME="${HF_HOME:-/Volumes/SSD/hf-cache}"
  export HF_HUB_DISABLE_PROGRESS_BARS="${HF_HUB_DISABLE_PROGRESS_BARS:-1}"
  run "$SCRIPT_DIR/check-benchmark-coverage.sh"
  run "$SCRIPT_DIR/check-hf-collections.sh"
  run "$SCRIPT_DIR/check-hf-model-cards.sh"
fi

if [[ "$RUN_DISTRIBUTED" == "1" ]]; then
  run "$SCRIPT_DIR/check-distributed-readiness.sh" "${caix_args[@]}"
fi

echo "publication gates ok"
