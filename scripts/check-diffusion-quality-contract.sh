#!/usr/bin/env bash
# Validate the no-load diffusion-quality prompt fixture and raw-evidence contract.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROMPTS="$REPO_DIR/quality/diffusion_prompts_v0.tsv"
VALIDATOR="$SCRIPT_DIR/diffusion-eval/validate-run.py"
README="$SCRIPT_DIR/diffusion-eval/README.md"
QUALITY_DOC="$REPO_DIR/docs/QUALITY_GATES.md"
API_CONTRACT="$REPO_DIR/quality/diffusion_api_contract_v0.json"

[[ -f "$PROMPTS" ]] || { echo "error: diffusion prompt fixture missing: $PROMPTS" >&2; exit 1; }
[[ -x "$VALIDATOR" || -f "$VALIDATOR" ]] || { echo "error: diffusion validator missing: $VALIDATOR" >&2; exit 1; }
[[ -f "$README" ]] || { echo "error: diffusion-eval README missing: $README" >&2; exit 1; }
[[ -f "$API_CONTRACT" ]] || { echo "error: diffusion API contract missing: $API_CONTRACT" >&2; exit 1; }

python3 "$VALIDATOR" --prompts "$PROMPTS" --prompts-only >/dev/null

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-diffusion-quality-contract.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT
run_dir="$tmpdir/raw-run"
mkdir -p "$run_dir"

cat > "$run_dir/metadata.json" <<'JSON'
{
  "caix_commit": "test",
  "model_repo": "redhillsmediafl/rhm-test-diffusion-caix",
  "model_revision": "test",
  "reference_path": "hf-reference",
  "candidate_bundle": "/tmp/diffusion-candidate.aimodel",
  "hardware": "test",
  "os_build": "test",
  "command": "synthetic diffusion quality contract check"
}
JSON

cat > "$run_dir/diffusion_api.json" <<'JSON'
{
  "selected_api_mode": "nonstreaming_only_v1",
  "streaming_decision": "streaming unsupported until committed-block SSE exists",
  "request_examples": [
    {"surface": "openai_chat_completions", "stream": false, "expected": "allowed after quality pass"},
    {"surface": "openai_chat_completions", "stream": true, "expected": "reject"}
  ],
  "pass": true
}
JSON

{
  printf 'prompt_id\tslice\treference_output\tcandidate_output\trubric_result\trater_notes\n'
  awk -F '\t' 'NR > 1 { print $1 "\t" $2 "\treference output\tcandidate output\tpass\tsynthetic contract row" }' "$PROMPTS"
} > "$run_dir/diffusion_quality.tsv"

cat > "$run_dir/summary.json" <<'JSON'
{
  "gate_ids": ["diffusion_block_quality", "diffusion_api_contract"],
  "aggregate_results": {
    "text_completion": {"passed": 10, "total": 10},
    "math_reasoning": {"passed": 10, "total": 10},
    "instruction_following": {"passed": 10, "total": 10}
  },
  "final_decision": "pass"
}
JSON

python3 "$VALIDATOR" --prompts "$PROMPTS" --run "$run_dir" >/dev/null

blocked_dir="$tmpdir/blocked-run"
cp -R "$run_dir" "$blocked_dir"
python3 - "$blocked_dir" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
quality_path = run_dir / "diffusion_quality.tsv"
with quality_path.open(newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
rows[0]["rubric_result"] = "fail"
rows[0]["rater_notes"] = "synthetic failed row"
with quality_path.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "prompt_id",
            "slice",
            "reference_output",
            "candidate_output",
            "rubric_result",
            "rater_notes",
        ],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(rows)

summary_path = run_dir / "summary.json"
summary = json.loads(summary_path.read_text())
summary["aggregate_results"]["text_completion"]["passed"] = 9
summary["final_decision"] = "blocked"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
PY

python3 "$VALIDATOR" --prompts "$PROMPTS" --run "$blocked_dir" >/dev/null

bad_dir="$tmpdir/bad-pass-run"
cp -R "$blocked_dir" "$bad_dir"
python3 - "$bad_dir/summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
summary = json.loads(path.read_text())
summary["final_decision"] = "pass"
path.write_text(json.dumps(summary, indent=2, sort_keys=True))
PY

if python3 "$VALIDATOR" --prompts "$PROMPTS" --run "$bad_dir" >"$tmpdir/bad-pass.out" 2>&1; then
  echo "error: diffusion run with failing quality row unexpectedly passed with final_decision=pass" >&2
  exit 1
fi
grep -F 'final_decision pass requires all diffusion quality rows to pass' "$tmpdir/bad-pass.out" >/dev/null \
  || { echo "error: diffusion bad-pass failure did not mention failing quality rows" >&2; cat "$tmpdir/bad-pass.out" >&2; exit 1; }

grep -F 'quality/diffusion_prompts_v0.tsv' "$README" >/dev/null \
  || { echo "error: diffusion-eval README must name the prompt fixture" >&2; exit 1; }
grep -F 'quality/raw/<run>' "$README" >/dev/null \
  || { echo "error: diffusion-eval README must name the raw evidence root" >&2; exit 1; }
grep -F 'quality/diffusion_prompts_v0.tsv' "$QUALITY_DOC" >/dev/null \
  || { echo "error: quality gates doc must name the diffusion prompt fixture" >&2; exit 1; }

echo "diffusion quality contract ok"
