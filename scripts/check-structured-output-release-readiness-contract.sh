#!/usr/bin/env bash
# Self-test structured-output release-readiness checks with local fixtures only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-structured-output-readiness.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

evidence="$tmpdir/smoke.json"
evidence_max_tokens="$tmpdir/smoke-max-tokens.json"
evidence_zero_tps="$tmpdir/smoke-zero-tps.json"
deps_exact="$tmpdir/deps-exact.tsv"
deps_branch="$tmpdir/deps-branch.tsv"

cat > "$evidence" <<'JSON'
{
  "backend": "PersistentModel",
  "decode_tokens_per_second": 55.0,
  "generated_token_count": 10,
  "model": "/tmp/qwen3-4b-coreai",
  "prompt": "Return a JSON object with one field named answer containing the word ok.",
  "prompt_token_count": 22,
  "schema": "{\"type\":\"object\",\"additionalProperties\":false,\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"]}",
  "stop_reason": "eos",
  "text": "{\"answer\":\"ok\"}"
}
JSON

cat > "$deps_exact" <<'TSV'
component	ecosystem	version_kind	version_or_branch	revision	location	evidence_source	notes
coreai-models	swiftpm	branch	main	1111111111111111111111111111111111111111	https://github.com/apple/coreai-models.git	Package.resolved	CoreAILM runtime package; development tracks main, release evidence records this SHA.
xgrammar	swiftpm	exact	v0.1.0	2222222222222222222222222222222222222222	https://github.com/mlc-ai/xgrammar	Package.resolved	Constrained decoding dependency in the caix Swift build graph.
TSV

"$SCRIPT_DIR/check-structured-output-release-readiness.sh" \
  --dependency-evidence "$deps_exact" \
  --evidence "$evidence" >/dev/null

python3 - "$evidence" "$evidence_max_tokens" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
payload["stop_reason"] = "maxTokens"
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True))
PY

if "$SCRIPT_DIR/check-structured-output-release-readiness.sh" \
    --dependency-evidence "$deps_exact" \
    --evidence "$evidence_max_tokens" >"$tmpdir/max-tokens.out" 2>&1
then
  echo "error: maxTokens structured-output evidence unexpectedly passed release readiness" >&2
  exit 1
fi
rg -q 'must stop by eos or stopSequence' "$tmpdir/max-tokens.out" || {
  echo "error: maxTokens failure did not mention complete stop requirement" >&2
  cat "$tmpdir/max-tokens.out" >&2
  exit 1
}

python3 - "$evidence" "$evidence_zero_tps" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text())
payload["decode_tokens_per_second"] = 0
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True))
PY

if "$SCRIPT_DIR/check-structured-output-release-readiness.sh" \
    --dependency-evidence "$deps_exact" \
    --evidence "$evidence_zero_tps" >"$tmpdir/zero-tps.out" 2>&1
then
  echo "error: zero-tps structured-output evidence unexpectedly passed release readiness" >&2
  exit 1
fi
rg -q 'positive decode_tokens_per_second' "$tmpdir/zero-tps.out" || {
  echo "error: zero-tps failure did not mention positive decode timing" >&2
  cat "$tmpdir/zero-tps.out" >&2
  exit 1
}

cat > "$deps_branch" <<'TSV'
component	ecosystem	version_kind	version_or_branch	revision	location	evidence_source	notes
coreai-models	swiftpm	branch	main	1111111111111111111111111111111111111111	https://github.com/apple/coreai-models.git	Package.resolved	CoreAILM runtime package; development tracks main, release evidence records this SHA.
xgrammar	swiftpm	branch	main	2222222222222222222222222222222222222222	https://github.com/mlc-ai/xgrammar	Package.resolved	Constrained decoding dependency in the caix Swift build graph.
TSV

if "$SCRIPT_DIR/check-structured-output-release-readiness.sh" \
    --dependency-evidence "$deps_branch" \
    --evidence "$evidence" >"$tmpdir/strict.out" 2>&1
then
  echo "error: branch-based xgrammar unexpectedly passed strict release readiness" >&2
  exit 1
fi
rg -q 'xgrammar is still a branch dependency' "$tmpdir/strict.out" || {
  echo "error: strict failure did not mention branch-based xgrammar" >&2
  cat "$tmpdir/strict.out" >&2
  exit 1
}

"$SCRIPT_DIR/check-structured-output-release-readiness.sh" \
  --allow-branch-dependencies \
  --dependency-evidence "$deps_branch" \
  --evidence "$evidence" >"$tmpdir/dev.out" 2>&1
rg -q 'development evidence only' "$tmpdir/dev.out" || {
  echo "error: development mode did not warn about branch-based xgrammar" >&2
  cat "$tmpdir/dev.out" >&2
  exit 1
}

echo "structured-output release readiness contract ok"
