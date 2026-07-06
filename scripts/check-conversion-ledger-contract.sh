#!/usr/bin/env bash
# Self-test conversion-ledger status contracts with local fixtures only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-conversion-ledger-contract.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

registry="$tmpdir/registry.json"
manifest="$tmpdir/MANIFEST.tsv"
ledger="$tmpdir/CONVERSION_LEDGER.tsv"

cat > "$registry" <<'JSON'
{
  "conversion_order": ["model-target", "model-draft", "model-qwen35"],
  "models": {
    "model-target": {"hf_repo": "example/model-target"},
    "model-draft": {"hf_repo": "example/model-draft"},
    "model-qwen35": {"hf_repo": "example/model-qwen35", "model_type": "qwen3_5", "context": 1048576}
  }
}
JSON

cat > "$manifest" <<'TSV'
repo	local_dir	kind	benchmark_mode	status	notes
redhillsmediafl/rhm-model-caix	model-coreai	standalone	decode	eligible	verified
redhillsmediafl/rhm-model-staged-caix	model-staged	staged	manual	component_only	staged manifest; hardware smoke required
redhillsmediafl/rhm-model-mtp-caix	model-mtp	mtp	eagle-mtp	eligible	benchmark with standalone target
redhillsmediafl/rhm-model-draft-caix	model-draft	draft	manual	component_only	component; benchmark with matching target
redhillsmediafl/rhm-qwen35-caix	qwen35-coreai	standalone	decode	eligible	full context with contiguous replay evidence
TSV

cat > "$ledger" <<'TSV'
model_key	source_repo	status	published_repo	next_step
model-target	example/model-target	published	redhillsmediafl/rhm-model-caix,redhillsmediafl/rhm-model-staged-caix,redhillsmediafl/rhm-model-mtp-caix	Standalone is benchmarked; staged package needs distributed hardware smoke; benchmark MTP against standalone target row.
model-draft	example/model-draft	component_published	redhillsmediafl/rhm-model-draft-caix	Keep as a component; benchmark only with the matching target package.
model-qwen35	example/model-qwen35	published	redhillsmediafl/rhm-qwen35-caix	Full-context contiguous replay gate passed; collect speed evidence.
TSV

"$SCRIPT_DIR/check-conversion-ledger.sh" \
  --registry "$registry" \
  --ledger "$ledger" \
  --manifest "$manifest" >/dev/null

bad_staged="$tmpdir/staged-missing.tsv"
cp "$ledger" "$bad_staged"
perl -0pi -e 's/Standalone is benchmarked; staged package needs distributed hardware smoke; benchmark MTP against standalone target row\./Standalone is benchmarked; benchmark MTP against standalone target row./' "$bad_staged"
if "$SCRIPT_DIR/check-conversion-ledger.sh" \
    --registry "$registry" \
    --ledger "$bad_staged" \
    --manifest "$manifest" >"$tmpdir/staged.out" 2>&1
then
  echo "error: staged published repo without hardware next_step unexpectedly passed" >&2
  exit 1
fi
grep -F 'staged published repo' "$tmpdir/staged.out" >/dev/null \
  || { echo "error: staged caveat failure did not mention staged next_step" >&2; cat "$tmpdir/staged.out" >&2; exit 1; }

bad_mtp="$tmpdir/mtp-missing.tsv"
cp "$ledger" "$bad_mtp"
perl -0pi -e 's/; benchmark MTP against standalone target row//' "$bad_mtp"
if "$SCRIPT_DIR/check-conversion-ledger.sh" \
    --registry "$registry" \
    --ledger "$bad_mtp" \
    --manifest "$manifest" >"$tmpdir/mtp.out" 2>&1
then
  echo "error: MTP published repo without target/draft next_step unexpectedly passed" >&2
  exit 1
fi
grep -F 'MTP/speculative published repo' "$tmpdir/mtp.out" >/dev/null \
  || { echo "error: MTP caveat failure did not mention target/draft next_step" >&2; cat "$tmpdir/mtp.out" >&2; exit 1; }

bad_draft="$tmpdir/draft-missing.tsv"
cp "$ledger" "$bad_draft"
perl -0pi -e 's/Keep as a component; benchmark only with the matching target package\./Published draft artifact./' "$bad_draft"
if "$SCRIPT_DIR/check-conversion-ledger.sh" \
    --registry "$registry" \
    --ledger "$bad_draft" \
    --manifest "$manifest" >"$tmpdir/draft.out" 2>&1
then
  echo "error: draft published repo without component next_step unexpectedly passed" >&2
  exit 1
fi
grep -F 'draft published repo' "$tmpdir/draft.out" >/dev/null \
  || { echo "error: draft caveat failure did not mention component next_step" >&2; cat "$tmpdir/draft.out" >&2; exit 1; }

bad_qwen35="$tmpdir/qwen35-missing.tsv"
cp "$ledger" "$bad_qwen35"
perl -0pi -e 's/Full-context contiguous replay gate passed; collect speed evidence[.]/Collect speed evidence./' "$bad_qwen35"
if "$SCRIPT_DIR/check-conversion-ledger.sh" \
    --registry "$registry" \
    --ledger "$bad_qwen35" \
    --manifest "$manifest" >"$tmpdir/qwen35.out" 2>&1
then
  echo "error: qwen3_5 published repo without replay/narrower-label next_step unexpectedly passed" >&2
  exit 1
fi
grep -F 'qwen3_5 published row' "$tmpdir/qwen35.out" >/dev/null \
  || { echo "error: qwen3_5 caveat failure did not mention replay/narrower-label evidence" >&2; cat "$tmpdir/qwen35.out" >&2; exit 1; }

echo "conversion ledger contract ok"
