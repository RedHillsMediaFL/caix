#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-publication-gates.sh [--hub] [--distributed] [--strict-benchmark-gaps] [--strict-evidence] [--caix <path>] [--brew-caix <path>]

Runs the non-heavy gates for publishing docs, cards, manifests, and benchmark evidence.
Default checks are local only. --hub adds Hugging Face metadata/model-card checks.
--distributed checks the Thunderbolt readiness gate and implies --strict-evidence.
--strict-benchmark-gaps fails when an eligible benchmark manifest row lacks raw evidence.
--strict-evidence requires tracked distributed evidence files.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_HUB=0
RUN_DISTRIBUTED=0
STRICT_BENCHMARK_GAPS=0
STRICT_EVIDENCE=0
caix_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hub) RUN_HUB=1; shift ;;
    --distributed) RUN_DISTRIBUTED=1; shift ;;
    --strict-benchmark-gaps) STRICT_BENCHMARK_GAPS=1; shift ;;
    --strict-evidence) STRICT_EVIDENCE=1; shift ;;
    --caix) caix_args+=(--caix "${2:?}"); shift 2 ;;
    --brew-caix) caix_args+=(--brew-caix "${2:?}"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) echo "error: unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$RUN_DISTRIBUTED" == "1" ]]; then
  STRICT_EVIDENCE=1
fi

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
if command -v ruby >/dev/null 2>&1; then
  run ruby -c "$REPO_DIR/Formula/caix.rb"
fi
run "$SCRIPT_DIR/check-token-handling.sh"
run "$SCRIPT_DIR/check-publication-gates-contract.sh"
run "$SCRIPT_DIR/check-public-copy-contract.sh"
run "$SCRIPT_DIR/check-structured-output-evidence-contract.sh"
run "$SCRIPT_DIR/check-structured-output-release-readiness-contract.sh"
run "$SCRIPT_DIR/check-public-copy.sh"
run "$SCRIPT_DIR/check-release-version-contract.sh"
run "$SCRIPT_DIR/check-version-sync.sh"
run "$SCRIPT_DIR/check-catalog-install-locks.sh"
run "$SCRIPT_DIR/check-disk-pressure-guard.sh"
run "$SCRIPT_DIR/check-dependency-evidence-contract.sh"
run "$SCRIPT_DIR/check-dependency-evidence.sh"
run "$SCRIPT_DIR/check-rdma-transport-contract.sh"
run "$SCRIPT_DIR/check-quant-eval-contract.sh"
run "$SCRIPT_DIR/check-diffusion-api-contract.sh"
run "$SCRIPT_DIR/check-diffusion-quality-contract.sh"
run "$SCRIPT_DIR/check-quality-gates.sh"
run "$SCRIPT_DIR/check-ceilings.sh"
run "$SCRIPT_DIR/check-qwen3-samelineage-export.sh"
run "$SCRIPT_DIR/check-hf-model-card-contract.sh"
run "$SCRIPT_DIR/check-hf-collections-contract.sh"
run "$SCRIPT_DIR/check-pipelined-kv-guardrail.sh"
run "$SCRIPT_DIR/check-benchmark-raw-contract.sh"
run "$SCRIPT_DIR/check-benchmark-gaps-contract.sh"
run "$SCRIPT_DIR/check-conversion-ledger-contract.sh"
run "$SCRIPT_DIR/check-tester-requests-contract.sh"
run "$SCRIPT_DIR/check-tester-requests.sh"
run "$SCRIPT_DIR/check-cleanup-safety.sh"
run "$SCRIPT_DIR/check-export-cleanliness-contract.sh"
run "$SCRIPT_DIR/check-conversion-guard.sh"
run "$SCRIPT_DIR/check-converter-cluster-metadata.sh"
run "$SCRIPT_DIR/check-conversion-ledger.sh"
run "$SCRIPT_DIR/check-conversion-gap-audit.sh"
run "$SCRIPT_DIR/check-cluster-plan.sh"
if [[ "$STRICT_EVIDENCE" == "1" ]]; then
  run "$SCRIPT_DIR/check-distributed-evidence-contract.sh" --require-tracked
else
  run "$SCRIPT_DIR/check-distributed-evidence-contract.sh"
fi
run "$SCRIPT_DIR/check-benchmark-raw.sh" --require-tracked
if [[ "$STRICT_BENCHMARK_GAPS" == "1" ]]; then
  run "$SCRIPT_DIR/check-benchmark-gaps.sh" --strict
else
  run "$SCRIPT_DIR/check-benchmark-gaps.sh"
fi
run "$SCRIPT_DIR/check-export-cleanliness.sh"

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
