#!/usr/bin/env bash
# Validate the no-load quant-eval task fixture and raw-evidence contract.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TASKS="$REPO_DIR/quality/quant_tasks_v0.tsv"
VALIDATOR="$SCRIPT_DIR/quant-eval/validate-run.py"
README="$SCRIPT_DIR/quant-eval/README.md"
QUALITY_DOC="$REPO_DIR/docs/QUALITY_GATES.md"

[[ -f "$TASKS" ]] || { echo "error: quant task fixture missing: $TASKS" >&2; exit 1; }
[[ -x "$VALIDATOR" || -f "$VALIDATOR" ]] || { echo "error: quant validator missing: $VALIDATOR" >&2; exit 1; }
[[ -f "$README" ]] || { echo "error: quant-eval README missing: $README" >&2; exit 1; }

python3 "$VALIDATOR" --tasks "$TASKS" --tasks-only >/dev/null

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-quant-eval-contract.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT
run_dir="$tmpdir/raw-run"
mkdir -p "$run_dir"

cat > "$run_dir/metadata.json" <<'JSON'
{
  "caix_commit": "test",
  "model_repo": "redhillsmediafl/rhm-test-caix",
  "model_revision": "test",
  "reference_bundle": "/tmp/reference.aimodel",
  "candidate_bundle": "/tmp/candidate.aimodel",
  "hardware": "test",
  "os_build": "test",
  "command": "synthetic contract check"
}
JSON

cat > "$run_dir/quant_ppl.json" <<'JSON'
{
  "token_count": 1000,
  "reference_perplexity": 10.0,
  "candidate_perplexity": 10.1,
  "relative_delta": 0.01,
  "pass": true
}
JSON

{
  printf 'prompt_id\tslice\texpected_answer\tcandidate_answer\tpass\n'
  awk -F '\t' 'NR > 1 { print $1 "\t" $2 "\t" $5 "\t" $5 "\ttrue" }' "$TASKS"
} > "$run_dir/quant_tasks.tsv"

cat > "$run_dir/summary.json" <<'JSON'
{
  "gate_ids": ["quant_ppl_default_4bit", "quant_task_eval"],
  "aggregate_scores": {
    "knowledge": {"passed": 40, "total": 40},
    "math_reasoning": {"passed": 20, "total": 20},
    "instruction_following": {"passed": 40, "total": 40}
  },
  "thresholds": {
    "quant_ppl_default_4bit": 0.02
  },
  "final_decision": "pass"
}
JSON

python3 "$VALIDATOR" --tasks "$TASKS" --run "$run_dir" >/dev/null

blocked_dir="$tmpdir/blocked-run"
cp -R "$run_dir" "$blocked_dir"
python3 - "$blocked_dir" <<'PY'
import csv
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
tasks_path = run_dir / "quant_tasks.tsv"
with tasks_path.open(newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
rows[0]["pass"] = "false"
with tasks_path.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["prompt_id", "slice", "expected_answer", "candidate_answer", "pass"],
        delimiter="\t",
    )
    writer.writeheader()
    writer.writerows(rows)

summary_path = run_dir / "summary.json"
summary = json.loads(summary_path.read_text())
summary["aggregate_scores"]["knowledge"]["passed"] = 39
summary["final_decision"] = "blocked"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
PY

python3 "$VALIDATOR" --tasks "$TASKS" --run "$blocked_dir" >/dev/null

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

if python3 "$VALIDATOR" --tasks "$TASKS" --run "$bad_dir" >"$tmpdir/bad-pass.out" 2>&1; then
  echo "error: quant run with failing task row unexpectedly passed with final_decision=pass" >&2
  exit 1
fi
grep -F 'final_decision pass requires all quant task rows to pass' "$tmpdir/bad-pass.out" >/dev/null \
  || { echo "error: quant bad-pass failure did not mention failing task rows" >&2; cat "$tmpdir/bad-pass.out" >&2; exit 1; }

grep -F 'quality/quant_tasks_v0.tsv' "$README" >/dev/null \
  || { echo "error: quant-eval README must name the task fixture" >&2; exit 1; }
grep -F 'quality/raw/<run>' "$README" >/dev/null \
  || { echo "error: quant-eval README must name the raw evidence root" >&2; exit 1; }
grep -F 'quality/quant_tasks_v0.tsv' "$QUALITY_DOC" >/dev/null \
  || { echo "error: quality gates doc must name the task fixture" >&2; exit 1; }

echo "quant eval contract ok"
