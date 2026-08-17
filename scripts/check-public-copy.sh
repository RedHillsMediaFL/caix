#!/usr/bin/env bash
# Fail public docs/pages that drift into benchmark placeholders, hype, or vague model-size language.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-public-copy.sh [path ...]

Default paths:
  README.md CHANGELOG.md docs web Formula

Checks public-facing text only. Do not point this at internal coordination logs.
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ "$#" -gt 0 ]; then
  paths=("$@")
else
  paths=(README.md CHANGELOG.md docs web Formula)
fi

existing=()
for path in "${paths[@]}"; do
  if [ -e "$path" ]; then
    existing+=("$path")
  fi
done

if [ "${#existing[@]}" -eq 0 ]; then
  echo "error: no input paths exist" >&2
  exit 2
fi

fail=0

scan() {
  local label="$1"
  local pattern="$2"
  if rg -n -i --glob '!benchmarks/raw/**' --glob '!benchmarks/reports/**' "$pattern" "${existing[@]}"; then
    echo "error: $label" >&2
    fail=1
  fi
}

scan_exact() {
  local label="$1"
  local pattern="$2"
  if rg -n --glob '!benchmarks/raw/**' --glob '!benchmarks/reports/**' "$pattern" "${existing[@]}"; then
    echo "error: $label" >&2
    fail=1
  fi
}

scan_structured_output_claims() {
  local matches filtered
  matches="$(
    rg -n -i --glob '!benchmarks/raw/**' --glob '!benchmarks/reports/**' \
      'supports?[^[:cntrl:].]*(response_format|json_schema|structured[- ]outputs?|constrained decoding)|(response_format|json_schema|structured[- ]outputs?|constrained decoding)[^[:cntrl:].]*(supported|enabled|available|ready|verified)' \
      "${existing[@]}" || true
  )"
  [[ -n "$matches" ]] || return 0

  filtered="$(
    printf '%s\n' "$matches" \
      | rg -v -i 'not supported|unsupported|until|before|requires?|required|no-load|code/typecheck|smoke|must not|does not' \
      || true
  )"
  [[ -n "$filtered" ]] || return 0

  printf '%s\n' "$filtered"
  echo "error: structured-output public claim requires real model smoke evidence" >&2
  fail=1
}

scan_prefix_cache_claims() {
  local matches filtered
  matches="$(
    rg -n -i --glob '!benchmarks/raw/**' --glob '!benchmarks/reports/**' \
      'prompt caching[^[:cntrl:].]*(supported|enabled|available|ready|verified|works?|automatic)|(prompt|prefix|context|kv)[- ]?(cache|caching|reuse|snapshot)[^[:cntrl:].]*(supported|enabled|available|ready|verified|automatic|cross[- ]session|conversation[- ]keyed|semantic|partial[- ]prefix|general|ssd[- ]persistent)|(semantic|partial[- ]prefix|cross[- ]session|conversation[- ]keyed|ssd[- ]persistent|general)[^[:cntrl:].]*(prompt|prefix|context|kv)[- ]?(cache|caching|reuse|snapshot)' \
      "${existing[@]}" || true
  )"
  [[ -n "$matches" ]] || return 0

  filtered="$(
    printf '%s\n' "$matches" \
      | rg -v -i 'not supported|unsupported|not available|no |future|deferred|until|before|requires?|required|gated?|gate|blocked|do not|must not|does not|only|exact[- ]continuation|audit(ed)?|observability|backlog|design|planned' \
      || true
  )"
  [[ -n "$filtered" ]] || return 0

  printf '%s\n' "$filtered"
  echo "error: prefix-cache public claim must be limited to exact continuation reuse on loaded CoreAILM fast handles" >&2
  fail=1
}

scan_multimodal_claims() {
  local matches filtered
  matches="$(
    rg -n -i --glob '!benchmarks/raw/**' --glob '!benchmarks/reports/**' \
      '(supports?|accepts?|serves?)[^[:cntrl:].]*(multimodal|vision|image[-/ ]?(input|request|content)|audio[-/ ]?(input|request|content)|video[-/ ]?(input|request|content))|(multimodal|vision|image[-/ ]?(input|request|content)|audio[-/ ]?(input|request|content)|video[-/ ]?(input|request|content))[^[:cntrl:].]*(supported|enabled|available|ready|verified|works?|accepted|served|serving|runtime bundle|runtime support)' \
      "${existing[@]}" || true
  )"
  [[ -n "$matches" ]] || return 0

  filtered="$(
    printf '%s\n' "$matches" \
      | rg -v -i 'not supported|unsupported|not available|no verified|no runtime|no-load|structural|preflight|fixture|parser|parses|parsed|reject(ed|s)?|returns? (a )?(clear )?400|http 400|until|before|requires?|required|gated?|gate|blocked|blocks? positive|admissible wording|public-copy guard|same guard|held|do not|must not|does not|no support|not yet|future|plan|planned|todo' \
      || true
  )"
  [[ -n "$filtered" ]] || return 0

  printf '%s\n' "$filtered"
  echo "error: multimodal public claim requires a verified runtime bundle and serving evidence" >&2
  fail=1
}

scan_serving_path_label_claims() {
  local matches filtered
  matches="$(
    rg -n -i --glob '!benchmarks/raw/**' --glob '!benchmarks/reports/**' \
      '(staged|same[- ]machine staged)[^[:cntrl:].]*(single[- ]device fast path|local fast path|one[- ]device fast path|fast local path|fastest local|faster on (one|a single) (mac|device))|(single[- ]device fast path|local fast path|one[- ]device fast path|fast local path|fastest local|faster on (one|a single) (mac|device))[^[:cntrl:].]*(staged|same[- ]machine staged)|(4[- ]?bit|4bit)[^[:cntrl:].]*(fp16[- ]?1:1|1:1[^[:cntrl:].]*fp16|hf[- ]?fp16[- ]?1:1)|(fp16[- ]?1:1|1:1[^[:cntrl:].]*fp16|hf[- ]?fp16[- ]?1:1)[^[:cntrl:].]*(4[- ]?bit|4bit)' \
      "${existing[@]}" || true
  )"
  [[ -n "$matches" ]] || return 0

  filtered="$(
    printf '%s\n' "$matches" \
      | rg -v -i 'not |must not|do not|does not|cannot|no |unsupported|blocked|held|future|requires?|required|per-model parity|internal directional|debug|correctness|label|guard|public-copy|admissible wording|single-device fast path, but|not a single-device fast path|4[- ]?bit bundles must not claim|must still distinguish' \
      || true
  )"
  [[ -n "$filtered" ]] || return 0

  printf '%s\n' "$filtered"
  echo "error: serving-path public copy must keep monolithic=single-device fast and staged=distributed; 4-bit bundles must not claim fp16 1:1" >&2
  fail=1
}

scan_monolithic_multimodal_claims() {
  local matches filtered
  matches="$(
    rg -n -i --glob '!benchmarks/raw/**' --glob '!benchmarks/reports/**' \
      '(monolithic|single[- ]device fast path|fused fast path)[^[:cntrl:].]*(multimodal|image[- ]?text|vision|image[-/ ]?(input|request|content))[^[:cntrl:].]*(supported|enabled|available|ready|verified|works?|accepted|served|serving|fast|buildable|shipping|ships?)|(multimodal|image[- ]?text|vision|image[-/ ]?(input|request|content))[^[:cntrl:].]*(monolithic|single[- ]device fast path|fused fast path)[^[:cntrl:].]*(supported|enabled|available|ready|verified|works?|accepted|served|serving|fast|buildable|shipping|ships?)|(supported|enabled|available|ready|verified|works?|accepted|served|serving|fast|buildable|shipping|ships?)[^[:cntrl:].]*(monolithic|single[- ]device fast path|fused fast path)[^[:cntrl:].]*(multimodal|image[- ]?text|vision|image[-/ ]?(input|request|content))' \
      "${existing[@]}" || true
  )"
  [[ -n "$matches" ]] || return 0

  filtered="$(
    printf '%s\n' "$matches" \
      | rg -v -i 'not |no |do not|must not|does not|cannot|without|until|before|requires?|required|future|planned|separate project|separate export|new export|new .*runtime contract|not a quick|staged[- ]only|current[^[:cntrl:].]*(path|handling)[^[:cntrl:].]*staged|depends on staged|public-copy|guard|claim rules?' \
      || true
  )"
  [[ -n "$filtered" ]] || return 0

  printf '%s\n' "$filtered"
  echo "error: monolithic multimodal public claims require a new export/runtime contract; current Gemma image-text serving is staged" >&2
  fail=1
}

scan_stateful_prefill_claims() {
  local matches filtered
  matches="$(
    rg -n -i --glob '!benchmarks/raw/**' --glob '!benchmarks/reports/**' \
      '(stateful|monolithic)[^[:cntrl:].]*(batch|wider|wide|full[- ]?batch|above 16|>16|over 16)[^[:cntrl:].]*prefill[^[:cntrl:].]*(safe|deterministic|correct|supported|enabled|available|ready|verified|release path|shipping|ships?|fast path)|(batch|wider|wide|full[- ]?batch|above 16|>16|over 16)[^[:cntrl:].]*prefill[^[:cntrl:].]*(stateful|monolithic)[^[:cntrl:].]*(safe|deterministic|correct|supported|enabled|available|ready|verified|release path|shipping|ships?|fast path)|(safe|deterministic|correct|supported|enabled|available|ready|verified|release path|shipping|ships?|fast path)[^[:cntrl:].]*(stateful|monolithic)[^[:cntrl:].]*(batch|wider|wide|full[- ]?batch|above 16|>16|over 16)[^[:cntrl:].]*prefill' \
      "${existing[@]}" || true
  )"
  [[ -n "$matches" ]] || return 0

  filtered="$(
    printf '%s\n' "$matches" \
      | rg -v -i 'not |no |do not|must not|does not|cannot|unsafe|nondeterministic|without|until|before|requires?|required|future|planned|blocked|gated?|gate|apple issue|upstream|chunk16|<=16|16 by default|not a release path|public-copy|guard|claim rules?|same gates?|determinism gates?' \
      || true
  )"
  [[ -n "$filtered" ]] || return 0

  printf '%s\n' "$filtered"
  echo "error: wider stateful monolithic prefill claims require deterministic >16-token evidence; chunk16 is the release-safe default" >&2
  fail=1
}

scan_speed_claims() {
  local matches filtered
  matches="$(
    rg -n -i --glob '!benchmarks/raw/**' --glob '!benchmarks/reports/**' \
      '(`?[0-9]+([.][0-9]+)?`?[[:space:]]*tok/s)|([0-9]+([.][0-9]+)?[[:space:]]*x[[:space:]]+(faster|slower|speedup))|(speedup[^[:cntrl:].]*[0-9]+([.][0-9]+)?[[:space:]]*x)' \
      "${existing[@]}" || true
  )"
  [[ -n "$matches" ]] || return 0

  filtered="$(
    printf '%s\n' "$matches" \
      | rg -v -i 'internal|not (a )?public|not publish|not publishable|do not publish|quarantine|quarantined|adjudication|comparator|ceiling|example|examples?|usage dashboard|live|rolling|decode tok/s|output tokens divided|benchmark rules|raw evidence|required evidence|public-copy guard|publication-gate fixture|contract fixture|token_accurate|directional|architecture decision input' \
      || true
  )"
  [[ -n "$filtered" ]] || return 0

  printf '%s\n' "$filtered"
  echo "error: public speed claims require publishable raw benchmark evidence or explicit internal/not-publishable framing" >&2
  fail=1
}

scan_distributed_readiness_claims() {
  local matches filtered
  matches="$(
    rg -n -i --glob '!benchmarks/raw/**' --glob '!benchmarks/reports/**' \
      '(distributed|staged|two[- ]machine|2[- ]machine|thunderbolt|macbook)[^[:cntrl:].]*(ready[- ]to[- ]test|ready for thunderbolt testing|upload[- ]?ready|upload readiness|ready to upload|publication ready)|(ready[- ]to[- ]test|ready for thunderbolt testing|upload[- ]?ready|upload readiness|ready to upload|publication ready)[^[:cntrl:].]*(distributed|staged|two[- ]machine|2[- ]machine|thunderbolt|macbook)' \
      "${existing[@]}" || true
  )"
  [[ -n "$matches" ]] || return 0

  filtered="$(
    printf '%s\n' "$matches" \
      | rg -v -i 'not |no |do not|must not|does not|cannot|blocked|deferred|later|future|until|before|requires?|required|needs?|gate|runbook|must print|prints?|not ready|check[- ]distributed[- ]readiness|publication policy|sign[- ]off|without|nothing about|no staged upload|no upload|current two-machine gate|hardware runbook' \
      || true
  )"
  [[ -n "$filtered" ]] || return 0

  printf '%s\n' "$filtered"
  echo "error: distributed readiness/upload claims require two-machine hardware evidence and sign-off" >&2
  fail=1
}

scan_hybrid_full_context_claims() {
  local matches filtered
  matches="$(
    rg -n -i --glob '!benchmarks/raw/**' --glob '!benchmarks/reports/**' \
      '(qwen3[_ .-]?5|qwen3[.]5|qwen3[.]6|qwythos)[^[:cntrl:].]*(full[- ]?1m|1m[- ]?(context|token)|1,048,576|1048576|1,000,000|million[- ]token|one[- ]million[- ]token|full[- ]native[- ]context)[^[:cntrl:].]*(verified|supported|enabled|available|ready|published|shipping|ships?|served|serves?|works?|claim)|(verified|supported|enabled|available|ready|published|shipping|ships?|served|serves?|works?)[^[:cntrl:].]*(qwen3[_ .-]?5|qwen3[.]5|qwen3[.]6|qwythos)[^[:cntrl:].]*(full[- ]?1m|1m[- ]?(context|token)|1,048,576|1048576|1,000,000|million[- ]token|one[- ]million[- ]token|full[- ]native[- ]context)' \
      "${existing[@]}" || true
  )"
  [[ -n "$matches" ]] || return 0

  filtered="$(
    printf '%s\n' "$matches" \
      | rg -v -i 'not |no |do not|must not|does not|cannot|without|until|before|requires?|required|needs?|gate|gated|blocked|deferred|future|bring[- ]up|unverified|too slow|narrow(ed|er)|public claim|claim requires|claim still needs|allocation|capacity only|kv capacity|contiguous replay|admissible wording|public-copy|guard' \
      || true
  )"
  [[ -n "$filtered" ]] || return 0

  printf '%s\n' "$filtered"
  echo "error: qwen3_5 full-context public claims require practical contiguous replay evidence or an approved narrower label" >&2
  fail=1
}

scan_gguf_multimodal_claims() {
  local matches filtered
  matches="$(
    rg -n -i --glob '!benchmarks/raw/**' --glob '!benchmarks/reports/**' \
      '((gguf|mmproj)[^[:cntrl:].]*(image[- ]?text|multimodal|vision|image[-/ ]?(input|request|content))[^[:cntrl:].]*(supported|enabled|available|ready|verified|works?|accepted|served|serving|converts?|conversion)|(image[- ]?text|multimodal|vision|image[-/ ]?(input|request|content))[^[:cntrl:].]*(gguf|mmproj)[^[:cntrl:].]*(supported|enabled|available|ready|verified|works?|accepted|served|serving|converts?|conversion)|(supported|enabled|available|ready|verified|works?|accepted|served|serving|converts?|conversion)[^[:cntrl:].]*(gguf|mmproj)[^[:cntrl:].]*(image[- ]?text|multimodal|vision|image[-/ ]?(input|request|content)))' \
      "${existing[@]}" || true
  )"
  [[ -n "$matches" ]] || return 0

  filtered="$(
    printf '%s\n' "$matches" \
      | rg -v -i 'not |no |do not|must not|does not|cannot|without|until|before|requires?|required|needs?|not yet|future|planned|blocked|gate|gated|candidate|text-only|safetensors|importer|mmproj[^[:cntrl:].]*(sidecar|files?).*(not|does not)|current[^[:cntrl:].]*(path|gguf)[^[:cntrl:].]*does not consume|gguf does not add architecture support|quality caveat' \
      || true
  )"
  [[ -n "$filtered" ]] || return 0

  printf '%s\n' "$filtered"
  echo "error: GGUF image-text public claims require a safetensors source or a verified mmproj importer" >&2
  fail=1
}

scan_rdma_claims() {
  local matches filtered
  matches="$(
    rg -n -i --glob '!benchmarks/raw/**' --glob '!benchmarks/reports/**' \
      '(rdma|tb5|thunderbolt 5|rdma_verbs_tb5|applethunderboltrdma|jaccl)[^[:cntrl:].]*(supported|enabled|available|ready|shipping|ships?|production|validated|verified|speedup|tensor[- ]parallel|all[- ]reduce)|(supported|enabled|available|ready|shipping|ships?|production|validated|verified|speedup|tensor[- ]parallel|all[- ]reduce)[^[:cntrl:].]*(rdma|tb5|thunderbolt 5|rdma_verbs_tb5|applethunderboltrdma|jaccl)' \
      "${existing[@]}" || true
  )"
  [[ -n "$matches" ]] || return 0

  filtered="$(
    printf '%s\n' "$matches" \
      | rg -v -i 'not |no |do not|must not|does not|cannot|without|until|before|requires?|required|future|planned|design|design-only|hardware[- ]gated|hardware evidence|gate|gated|blocked|deferred|later|fallback|contract|claim rules?|claims? require|public-copy|guard|current fleet cannot|current shipping|tcp_worker_frame|source-backed|interface|stub|stubbed|transport swap|negotiat(e|ion)|capabilit(y|ies)|admissible wording|DTS|when a tb5 pair|when tb5 hardware|optional|if/when|not testable' \
      || true
  )"
  [[ -n "$filtered" ]] || return 0

  printf '%s\n' "$filtered"
  echo "error: RDMA/TB5 public claims require TB5 hardware evidence, negotiation evidence, token-accurate raw evidence, and sign-off" >&2
  fail=1
}

scan "benchmark placeholder or unsupported public speed claim" \
  'benchmark pending|fastest|blazing|guaranteed|100%[[:space:]]+(compatible|support(ed)?|coverage|accurate|accuracy|working|faster|speed|safe|verified)'
scan "raw benchmark speed number in public copy" \
  '[0-9]+([.][0-9]+)?[[:space:]]*tok/s'
scan_speed_claims
scan_structured_output_claims
scan_prefix_cache_claims
scan_multimodal_claims
scan_serving_path_label_claims
scan_monolithic_multimodal_claims
scan_stateful_prefill_claims
scan_distributed_readiness_claims
scan_hybrid_full_context_claims
scan_gguf_multimodal_claims
scan_rdma_claims
scan "marketing/hype wording" \
  'revolutionary|game[- ]chang(er|ing)|world[- ]class|best[- ]in[- ]class|cutting[- ]edge|state[- ]of[- ]the[- ]art|next[- ]gen|breakthrough|magic|seamless|effortless|supercharge|turbocharge|unleash|gimmick'
scan "support gimmick wording" \
  'donat(e|ion)?|sticker'
scan_exact "uppercase caix brand wording" \
  '\bC[A]IX\b'
scan "vague model-size wording" \
  'large model|large chat|dense large|\blargest\b'
scan "bare large wording; use parameter count, disk size, memory, and license instead" \
  '\blarge\b'
scan "unsafe export cleanup command; use scripts/remove-export.sh" \
  'rm[[:space:]]+-rf[^[:cntrl:]]*models/exports'

if [ "$fail" -ne 0 ]; then
  exit 1
fi

echo "public copy ok"
