#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EVIDENCE_DIR="$REPO_DIR/docs/distributed-evidence"
PROMPT_SET="docs/distributed-evidence/qwen3-0.6b-prompts.txt"
README="$EVIDENCE_DIR/README.md"
TEACHER_FORCED_EVIDENCE="$EVIDENCE_DIR/qwen3-0.6b-teacher-forced-fp16-128.txt"
EXPECTED_PROMPTS=8
REQUIRE_TRACKED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --require-tracked) REQUIRE_TRACKED=1; shift ;;
    -h|--help)
      cat <<'USAGE'
Usage: scripts/check-distributed-evidence-contract.sh [--require-tracked]

Validates the no-load distributed evidence contract. Use --require-tracked for final publication
review so local scratch evidence cannot satisfy the contract.
USAGE
      exit 0
      ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

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

require_tracked_path() {
  local label="$1"
  local value="$2"
  git -C "$REPO_DIR" ls-files --error-unmatch -- "$value" >/dev/null 2>&1 \
    || fail "$label path is not tracked: $value"
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
  require_tracked_path "$label" "$value"
}

check_repo_path prompt_set "$PROMPT_SET"
[[ "$(prompt_count "$REPO_DIR/$PROMPT_SET")" == "$EXPECTED_PROMPTS" ]] \
  || fail "$PROMPT_SET must contain $EXPECTED_PROMPTS non-empty prompts"
grep -Fq "prompt_set=$PROMPT_SET" "$README" \
  || fail "README missing prompt_set contract"
grep -Fq "qwen3-0.6b-teacher-forced-fp16-128.txt" "$README" \
  || fail "README missing teacher-forced parity evidence contract"

check_teacher_forced_evidence() {
  local file="$1"
  local rel="${file#$REPO_DIR/}"
  [[ -s "$file" ]] || fail "teacher-forced parity evidence is missing: $rel"
  if [[ "$REQUIRE_TRACKED" == "1" ]]; then
    require_tracked_path "$rel" "$rel"
  fi

  local result mode model oracle caix_commit coreai_models_commit prompt_set prompts max_tokens
  local temperature token_match teacher_forced tie_tolerance exact_matches accepted_ties
  local real_divergences upload_ready raw_log
  result="$(evidence_value result "$file")"
  mode="$(evidence_value mode "$file")"
  model="$(evidence_value model "$file")"
  oracle="$(evidence_value oracle "$file")"
  caix_commit="$(evidence_value caix_commit "$file")"
  coreai_models_commit="$(evidence_value coreai_models_commit "$file")"
  prompt_set="$(evidence_value prompt_set "$file")"
  prompts="$(evidence_value prompts "$file")"
  max_tokens="$(evidence_value max_tokens "$file")"
  temperature="$(evidence_value temperature "$file")"
  token_match="$(evidence_value token_match "$file")"
  teacher_forced="$(evidence_value teacher_forced "$file")"
  tie_tolerance="$(evidence_value tie_tolerance "$file")"
  exact_matches="$(evidence_value exact_matches "$file")"
  accepted_ties="$(evidence_value accepted_ties "$file")"
  real_divergences="$(evidence_value real_divergences "$file")"
  upload_ready="$(evidence_value upload_ready "$file")"
  raw_log="$(evidence_value raw_log "$file")"

  [[ "$result" == "pass" ]] || fail "$rel result must be pass"
  [[ "$mode" == "same-machine-teacher-forced" ]] || fail "$rel mode must be same-machine-teacher-forced"
  [[ "$model" == "qwen3-0.6b-coreai-staged-f16-noopt-first3" ]] || fail "$rel model is unexpected: $model"
  [[ "$oracle" == "hf-fp16" ]] || fail "$rel oracle must be hf-fp16"
  [[ "$caix_commit" =~ ^[0-9a-f]{40}$ ]] || fail "$rel caix_commit must be a 40-character SHA"
  git -C "$REPO_DIR" cat-file -e "$caix_commit^{commit}" 2>/dev/null \
    || fail "$rel caix_commit is not present in this repository: $caix_commit"
  [[ "$coreai_models_commit" =~ ^[0-9a-f]{40}$ ]] \
    || fail "$rel coreai_models_commit must be a 40-character SHA"
  [[ "$prompt_set" == "$PROMPT_SET" ]] || fail "$rel prompt_set must be $PROMPT_SET"
  [[ "$prompts" == "$EXPECTED_PROMPTS" ]] || fail "$rel prompts must be $EXPECTED_PROMPTS"
  [[ "$(prompt_count "$REPO_DIR/$prompt_set")" == "$prompts" ]] \
    || fail "$rel prompts=$prompts does not match prompt_set line count"
  [[ "$max_tokens" == "128" ]] || fail "$rel max_tokens must be 128"
  [[ "$temperature" == "0" ]] || fail "$rel temperature must be 0"
  [[ "$token_match" == "tie-aware" ]] || fail "$rel token_match must be tie-aware"
  [[ "$teacher_forced" == "true" ]] || fail "$rel teacher_forced must be true"
  [[ "$tie_tolerance" == "0.02" ]] || fail "$rel tie_tolerance must be 0.02"
  [[ "$exact_matches" == "1020" ]] || fail "$rel exact_matches must be 1020"
  [[ "$accepted_ties" == "4" ]] || fail "$rel accepted_ties must be 4"
  [[ "$real_divergences" == "0" ]] || fail "$rel real_divergences must be 0"
  [[ "$upload_ready" == "false" ]] || fail "$rel upload_ready must be false"
  [[ "$raw_log" == docs/distributed-evidence/* && "$raw_log" != *..* ]] \
    || fail "$rel raw_log must be a repo-relative distributed-evidence path"
  [[ -s "$REPO_DIR/$raw_log" ]] || fail "$rel raw_log path is missing or empty: $raw_log"
  if [[ "$REQUIRE_TRACKED" == "1" ]]; then
    require_tracked_path "$rel raw_log" "$raw_log"
  fi
  grep -Fq "Executed 1 test, with 0 failures" "$REPO_DIR/$raw_log" \
    || fail "$raw_log missing XCTest pass summary"
  [[ "$(grep -cF '[caix-token-tie]' "$REPO_DIR/$raw_log")" == "$accepted_ties" ]] \
    || fail "$raw_log tie count does not match accepted_ties"
}

check_teacher_forced_evidence "$TEACHER_FORCED_EVIDENCE"

for file in \
  "$EVIDENCE_DIR/same-machine-qwen3-0.6b-token-match.txt" \
  "$EVIDENCE_DIR/loopback-qwen3-0.6b-token-match.txt"; do
  [[ -s "$file" ]] || continue
  rel="${file#$REPO_DIR/}"
  prompt_set="$(evidence_value prompt_set "$file")"
  prompts="$(evidence_value prompts "$file")"
  asset_digests="$(evidence_value asset_digests "$file")"
  [[ "$prompts" =~ ^[1-9][0-9]*$ ]] || fail "$rel prompts must be positive"
  [[ "$prompt_set" == "$PROMPT_SET" ]] || fail "$rel prompt_set must be $PROMPT_SET"
  check_repo_path "$rel prompt_set" "$prompt_set"
  [[ "$(prompt_count "$REPO_DIR/$prompt_set")" == "$prompts" ]] \
    || fail "$rel prompts=$prompts does not match prompt_set line count"
  check_repo_path "$rel asset_digests" "$asset_digests"
done

echo "distributed evidence contract ok"
