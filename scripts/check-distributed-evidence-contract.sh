#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EVIDENCE_DIR="$REPO_DIR/docs/distributed-evidence"
PROMPT_SET="docs/distributed-evidence/qwen3-0.6b-prompts.txt"
README="$EVIDENCE_DIR/README.md"
EXPECTED_PROMPTS=8

fail() {
  echo "error: $*" >&2
  exit 1
}

evidence_value() {
  local key="$1"
  local file="$2"
  awk -v key="$key" '
    index($0, key "=") == 1 {
      sub("^[^=]*=", "")
      print
      exit
    }
  ' "$file"
}

prompt_count() {
  local path="$1"
  awk 'NF { count += 1 } END { print count + 0 }' "$path"
}

check_repo_path() {
  local label="$1"
  local value="$2"
  [[ -n "$value" ]] || fail "$label is missing"
  [[ "$value" != /* && "$value" != *://* &&
     "$value" != "." && "$value" != ".." &&
     "$value" != ../* && "$value" != */../* && "$value" != */.. ]] \
    || fail "$label must be a repo-relative path: $value"
  [[ -e "$REPO_DIR/$value" ]] || fail "$label path is missing: $value"
  git -C "$REPO_DIR" ls-files --error-unmatch -- "$value" >/dev/null 2>&1 \
    || fail "$label path is not tracked: $value"
}

check_repo_path prompt_set "$PROMPT_SET"
[[ "$(prompt_count "$REPO_DIR/$PROMPT_SET")" == "$EXPECTED_PROMPTS" ]] \
  || fail "$PROMPT_SET must contain $EXPECTED_PROMPTS non-empty prompts"
grep -Fq "prompt_set=$PROMPT_SET" "$README" \
  || fail "README missing prompt_set contract"

for file in \
  "$EVIDENCE_DIR/same-machine-qwen3-0.6b-token-match.txt" \
  "$EVIDENCE_DIR/loopback-qwen3-0.6b-token-match.txt"; do
  [[ -s "$file" ]] || continue
  rel="${file#$REPO_DIR/}"
  prompt_set="$(evidence_value prompt_set "$file")"
  prompts="$(evidence_value prompts "$file")"
  [[ "$prompts" =~ ^[1-9][0-9]*$ ]] || fail "$rel prompts must be positive"
  [[ "$prompt_set" == "$PROMPT_SET" ]] || fail "$rel prompt_set must be $PROMPT_SET"
  check_repo_path "$rel prompt_set" "$prompt_set"
  [[ "$(prompt_count "$REPO_DIR/$prompt_set")" == "$prompts" ]] \
    || fail "$rel prompts=$prompts does not match prompt_set line count"
done

echo "distributed evidence contract ok"
