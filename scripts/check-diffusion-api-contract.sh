#!/usr/bin/env bash
# Validate the no-load diffusion API contract.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTRACT="$REPO_DIR/quality/diffusion_api_contract_v0.json"
DOC="$REPO_DIR/docs/DIFFUSION_API.md"
QUALITY_DOC="$REPO_DIR/docs/QUALITY_GATES.md"
QUALITY_MANIFEST="$REPO_DIR/benchmarks/QUALITY_GATES.tsv"

[[ -f "$CONTRACT" ]] || { echo "error: diffusion API contract missing: $CONTRACT" >&2; exit 1; }
[[ -f "$DOC" ]] || { echo "error: diffusion API doc missing: $DOC" >&2; exit 1; }

python3 - "$CONTRACT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open() as handle:
    contract = json.load(handle)

def require(condition, message):
    if not condition:
        print(f"error: {message}", file=sys.stderr)
        raise SystemExit(1)

require(contract.get("schema") == "caix.diffusion_api_contract.v0", "unexpected diffusion contract schema")
require(contract.get("selected_api_mode") == "nonstreaming_only_v1", "diffusion v1 must stay non-streaming-only")
request = contract.get("request_contract", {})
for key in [
    "openai_chat_completions_nonstreaming",
    "openai_chat_completions_stream",
    "anthropic_messages_nonstreaming",
    "anthropic_messages_stream",
]:
    require(key in request, f"request_contract missing {key}")
require(request["openai_chat_completions_stream"].startswith("reject_"), "OpenAI streaming must reject for diffusion v1")
require(request["anthropic_messages_stream"].startswith("reject_"), "Anthropic streaming must reject for diffusion v1")

streaming = contract.get("streaming_contract", {})
require(streaming.get("token_delta_streaming") is False, "diffusion token-delta streaming must be false")
require(streaming.get("committed_block_sse") == "deferred", "committed-block SSE must remain deferred until implemented")
minimum_fields = set(streaming.get("minimum_block_payload_fields", []))
for field in ["block_index", "text", "accepted_token_count", "denoise_step_count", "stop_reason"]:
    require(field in minimum_fields, f"committed-block SSE field missing: {field}")

rejection = contract.get("required_rejection", {})
require(rejection.get("http_status") == 400, "streaming rejection status must be 400")
message_terms = set(rejection.get("message_must_contain", []))
for term in ["block diffusion", "streaming", "not supported"]:
    require(term in message_terms, f"streaming rejection term missing: {term}")

artifacts = set(contract.get("real_run_artifacts", []))
for artifact in [
    "quality/raw/<run>/metadata.json",
    "quality/raw/<run>/diffusion_quality.tsv",
    "quality/raw/<run>/diffusion_api.json",
    "quality/raw/<run>/summary.json",
]:
    require(artifact in artifacts, f"real run artifact missing: {artifact}")
PY

grep -F 'non-streaming-only' "$DOC" >/dev/null \
  || { echo "error: diffusion API doc must state non-streaming-only" >&2; exit 1; }
grep -F 'quality/diffusion_api_contract_v0.json' "$DOC" >/dev/null \
  || { echo "error: diffusion API doc must link the contract JSON" >&2; exit 1; }
grep -F 'Token-delta streaming is not a valid diffusion claim.' "$DOC" >/dev/null \
  || { echo "error: diffusion API doc must reject token-delta streaming claims" >&2; exit 1; }
grep -F 'quality/diffusion_api_contract_v0.json' "$QUALITY_DOC" >/dev/null \
  || { echo "error: quality gates doc must link the diffusion API contract" >&2; exit 1; }
awk -F '\t' '$1 == "diffusion_api_contract" && $7 == "nonstreaming_only_v1_documented" { found = 1 } END { exit found ? 0 : 1 }' "$QUALITY_MANIFEST" \
  || { echo "error: diffusion_api_contract gate must pin nonstreaming_only_v1_documented" >&2; exit 1; }

echo "diffusion api contract ok"
