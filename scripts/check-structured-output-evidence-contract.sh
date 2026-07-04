#!/usr/bin/env bash
# Self-test structured-output evidence validation with local fixtures only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-structured-output-evidence.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

valid="$tmpdir/valid.json"
invalid_text="$tmpdir/invalid-text.json"
invalid_extra="$tmpdir/invalid-extra.json"

cat > "$valid" <<'JSON'
{
  "backend": "PersistentModel",
  "decode_tokens_per_second": 55.0,
  "generated_token_count": 10,
  "model": "/tmp/qwen3-4b-coreai",
  "prompt": "Return a JSON object with one field named answer containing the word ok.",
  "prompt_token_count": 22,
  "schema": "{\"type\":\"object\",\"additionalProperties\":false,\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"]}",
  "stop_reason": "eos",
  "text": "{\n  \"answer\": \"ok\"\n}"
}
JSON

"$SCRIPT_DIR/check-structured-output-evidence.sh" "$valid" >/dev/null

cat > "$invalid_text" <<'JSON'
{
  "backend": "PersistentModel",
  "decode_tokens_per_second": 55.0,
  "generated_token_count": 10,
  "model": "/tmp/qwen3-4b-coreai",
  "prompt": "Return a JSON object with one field named answer containing the word ok.",
  "prompt_token_count": 22,
  "schema": "{\"type\":\"object\",\"additionalProperties\":false,\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"]}",
  "stop_reason": "eos",
  "text": "not json"
}
JSON

if "$SCRIPT_DIR/check-structured-output-evidence.sh" "$invalid_text" >"$tmpdir/invalid-text.out" 2>&1; then
  echo "error: invalid generated JSON fixture unexpectedly passed" >&2
  exit 1
fi
rg -q '`text` is not valid JSON' "$tmpdir/invalid-text.out" || {
  echo "error: invalid generated JSON failure did not mention text JSON" >&2
  cat "$tmpdir/invalid-text.out" >&2
  exit 1
}

cat > "$invalid_extra" <<'JSON'
{
  "backend": "PersistentModel",
  "decode_tokens_per_second": 55.0,
  "generated_token_count": 10,
  "model": "/tmp/qwen3-4b-coreai",
  "prompt": "Return a JSON object with one field named answer containing the word ok.",
  "prompt_token_count": 22,
  "schema": "{\"type\":\"object\",\"additionalProperties\":false,\"properties\":{\"answer\":{\"type\":\"string\"}},\"required\":[\"answer\"]}",
  "stop_reason": "eos",
  "text": "{\"answer\":\"ok\",\"extra\":\"bad\"}"
}
JSON

if "$SCRIPT_DIR/check-structured-output-evidence.sh" "$invalid_extra" >"$tmpdir/invalid-extra.out" 2>&1; then
  echo "error: schema-violating generated JSON fixture unexpectedly passed" >&2
  exit 1
fi
rg -q 'extra keys not allowed by schema' "$tmpdir/invalid-extra.out" || {
  echo "error: schema violation failure did not mention extra keys" >&2
  cat "$tmpdir/invalid-extra.out" >&2
  exit 1
}

echo "structured-output evidence contract ok"
