#!/usr/bin/env bash
# Validate no-load structured-output smoke evidence JSON.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-structured-output-evidence.sh <evidence.json> [...]

Validates JSON artifacts written by scripts/check-structured-output-smoke.sh.
This does not build, load models, run inference, contact the network, or publish claims.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -eq 0 ]]; then
  usage >&2
  exit 2
fi

python3 - "$@" <<'PY'
import json
import math
import sys
from pathlib import Path


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def load_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        fail(f"{path}: invalid JSON: {exc}")


def require_string(obj: dict, key: str, path: Path) -> str:
    value = obj.get(key)
    require(isinstance(value, str) and value.strip(), f"{path}: `{key}` must be a non-empty string")
    return value


def require_positive_int(obj: dict, key: str, path: Path) -> int:
    value = obj.get(key)
    require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"{path}: `{key}` must be a positive integer")
    return value


def require_nonnegative_number(obj: dict, key: str, path: Path) -> float:
    value = obj.get(key)
    require(isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{path}: `{key}` must be numeric")
    value = float(value)
    require(math.isfinite(value) and value >= 0, f"{path}: `{key}` must be finite and non-negative")
    return value


def validate_schema_subset(schema: dict, generated, path: Path) -> None:
    require(schema.get("type") == "object", f"{path}: schema type must be object")
    require(isinstance(generated, dict), f"{path}: generated text must parse as a JSON object")

    required = schema.get("required", [])
    require(isinstance(required, list) and all(isinstance(item, str) for item in required),
            f"{path}: schema `required` must be a string array")
    for key in required:
        require(key in generated, f"{path}: generated JSON missing required key `{key}`")

    properties = schema.get("properties", {})
    require(isinstance(properties, dict), f"{path}: schema `properties` must be an object")
    if schema.get("additionalProperties") is False:
        extra = sorted(set(generated) - set(properties))
        require(not extra, f"{path}: generated JSON has extra keys not allowed by schema: {', '.join(extra)}")

    for key, rule in properties.items():
        if key not in generated:
            continue
        require(isinstance(rule, dict), f"{path}: schema property `{key}` must be an object")
        expected_type = rule.get("type")
        if expected_type == "string":
            require(isinstance(generated[key], str), f"{path}: generated key `{key}` must be a string")
        elif expected_type == "number":
            require(isinstance(generated[key], (int, float)) and not isinstance(generated[key], bool),
                    f"{path}: generated key `{key}` must be numeric")
        elif expected_type == "integer":
            require(isinstance(generated[key], int) and not isinstance(generated[key], bool),
                    f"{path}: generated key `{key}` must be an integer")
        elif expected_type == "boolean":
            require(isinstance(generated[key], bool), f"{path}: generated key `{key}` must be a boolean")
        elif expected_type == "object":
            require(isinstance(generated[key], dict), f"{path}: generated key `{key}` must be an object")
        elif expected_type == "array":
            require(isinstance(generated[key], list), f"{path}: generated key `{key}` must be an array")


allowed_stops = {"eos", "maxTokens", "contextLimit", "stopSequence"}
required_keys = {
    "backend",
    "model",
    "prompt",
    "schema",
    "text",
    "prompt_token_count",
    "generated_token_count",
    "stop_reason",
    "decode_tokens_per_second",
}

for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    require(path.is_file(), f"structured-output evidence file not found: {path}")
    payload = load_json(path)
    require(isinstance(payload, dict), f"{path}: evidence must be a JSON object")
    missing = sorted(required_keys - set(payload))
    require(not missing, f"{path}: evidence missing required keys: {', '.join(missing)}")

    backend = require_string(payload, "backend", path)
    require(backend == "PersistentModel", f"{path}: backend must be PersistentModel")
    require_string(payload, "model", path)
    require_string(payload, "prompt", path)
    stop_reason = require_string(payload, "stop_reason", path)
    require(stop_reason in allowed_stops, f"{path}: unexpected stop_reason `{stop_reason}`")
    require_positive_int(payload, "prompt_token_count", path)
    require_positive_int(payload, "generated_token_count", path)
    require_nonnegative_number(payload, "decode_tokens_per_second", path)

    schema_text = require_string(payload, "schema", path)
    try:
        schema = json.loads(schema_text)
    except json.JSONDecodeError as exc:
        fail(f"{path}: `schema` is not valid JSON: {exc}")
    require(isinstance(schema, dict), f"{path}: `schema` must decode to a JSON object")

    output_text = require_string(payload, "text", path).strip()
    try:
        generated = json.loads(output_text)
    except json.JSONDecodeError as exc:
        fail(f"{path}: `text` is not valid JSON: {exc}")
    validate_schema_subset(schema, generated, path)

print(f"structured-output evidence ok ({len(sys.argv) - 1} file(s))")
PY
