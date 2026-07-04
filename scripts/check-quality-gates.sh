#!/usr/bin/env bash
# Validate the no-load quality gate specification for quant and diffusion promotion.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-quality-gates.sh [--manifest <path>] [--doc <path>]

Checks the static quant/diffusion quality-gate manifest and documentation.
Does not load models, run evaluations, benchmark, download, or contact external services.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$REPO_DIR/benchmarks/QUALITY_GATES.tsv"
DOC="$REPO_DIR/docs/QUALITY_GATES.md"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="${2:?}"; shift 2 ;;
    --doc) DOC="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$MANIFEST" ]] || { echo "error: quality gate manifest not found: $MANIFEST" >&2; exit 2; }
[[ -f "$DOC" ]] || { echo "error: quality gate doc not found: $DOC" >&2; exit 2; }

expected_header=$'gate_id\tphase\tscope\treference_backend\tcandidate_backend\tmetric\tthreshold\trequired_slices\tevidence_dir\tpublish_rule\tnotes'
actual_header="$(head -n 1 "$MANIFEST")"
if [[ "$actual_header" != "$expected_header" ]]; then
  echo "error: unexpected QUALITY_GATES.tsv header" >&2
  echo "expected: $expected_header" >&2
  echo "actual:   $actual_header" >&2
  exit 1
fi

awk -F '\t' '
  NR == 1 { next }
  NF != 11 {
    printf "error: %s:%d has %d fields, expected 11\n", FILENAME, NR, NF > "/dev/stderr"
    bad = 1
    next
  }
  $1 == "" || $2 == "" || $3 == "" || $4 == "" || $5 == "" || $6 == "" || $7 == "" || $8 == "" || $9 == "" || $10 == "" {
    printf "error: %s:%d has an empty required field\n", FILENAME, NR > "/dev/stderr"
    bad = 1
  }
  $9 !~ /^quality\/raw\/<run>\// {
    printf "error: %s:%d evidence_dir must live under quality/raw/<run>/\n", FILENAME, NR > "/dev/stderr"
    bad = 1
  }
  $2 != "P2.4" && $2 != "P3" {
    printf "error: %s:%d phase must be P2.4 or P3\n", FILENAME, NR > "/dev/stderr"
    bad = 1
  }
  { seen[$1] = 1 }
  END {
    split("quant_ppl_default_4bit quant_ppl_ladder_variant quant_task_eval diffusion_block_quality diffusion_api_contract", required, " ")
    for (i in required) {
      if (!seen[required[i]]) {
        printf "error: missing required quality gate row: %s\n", required[i] > "/dev/stderr"
        bad = 1
      }
    }
    exit bad ? 1 : 0
  }
' "$MANIFEST"

require_manifest() {
  local pattern="$1"
  local message="$2"
  if ! awk -F '\t' -v pat="$pattern" 'NR > 1 && $0 ~ pat { found = 1 } END { exit found ? 0 : 1 }' "$MANIFEST"; then
    echo "error: $message" >&2
    exit 1
  fi
}

require_doc() {
  local pattern="$1"
  local message="$2"
  if ! grep -F "$pattern" "$DOC" >/dev/null; then
    echo "error: $message" >&2
    exit 1
  fi
}

require_manifest '^quant_task_eval\t.*math_reasoning' "quant task gate must require a math_reasoning slice"
require_manifest '^diffusion_block_quality\t.*math_reasoning' "diffusion quality gate must require a math_reasoning slice"
require_manifest '^quant_ppl_default_4bit\t.*<=0.02' "default 4-bit perplexity gate must keep the <=0.02 threshold"
require_manifest '^quant_ppl_ladder_variant\t.*<=0.05' "ladder perplexity gate must keep the <=0.05 threshold"
require_manifest '^diffusion_api_contract\t.*nonstreaming_only_v1_documented' "diffusion API gate must pin non-streaming-only v1"

require_doc 'quant_ppl_default_4bit' "quality doc must describe quant_ppl_default_4bit"
require_doc 'quant_ppl_ladder_variant' "quality doc must describe quant_ppl_ladder_variant"
require_doc 'quant_task_eval' "quality doc must describe quant_task_eval"
require_doc 'diffusion_block_quality' "quality doc must describe diffusion_block_quality"
require_doc 'diffusion_api_contract' "quality doc must describe diffusion_api_contract"
require_doc 'math_reasoning' "quality doc must require a math_reasoning slice"
require_doc 'quality/raw/<run>/metadata.json' "quality doc must define raw metadata artifact"
require_doc 'Do not use the pipelined text path for perplexity' "quality doc must preserve the sequential-logit evaluator constraint"
require_doc 'quality/diffusion_api_contract_v0.json' "quality doc must link the diffusion API contract"
require_doc 'quality/diffusion_prompts_v0.tsv' "quality doc must link the diffusion prompt fixture"

echo "quality gates ok"
