#!/usr/bin/env bash
set -euo pipefail

MODEL=${CAIX_PREFILL_MITIGATION_MODEL:?set CAIX_PREFILL_MITIGATION_MODEL to a monolithic caix bundle}
PROMPT=${CAIX_PREFILL_MITIGATION_PROMPT:?set CAIX_PREFILL_MITIGATION_PROMPT to a >16-token prompt file}
RUNS=${CAIX_PREFILL_DETERMINISM_RUNS:-3}
MAX_TOKENS=${CAIX_PREFILL_MITIGATION_MAX_TOKENS:-8}
OUT=${CAIX_PREFILL_DETERMINISM_OUT:-.tmp/monolithic-prefill-determinism}

mkdir -p "$OUT"
rm -f "$OUT"/run*.tokens "$OUT"/run*.meta "$OUT"/run*.log

for run in $(seq 1 "$RUNS"); do
  COREAI_RUNTIME=1 \
  CAIX_PREFILL_MITIGATION_MODEL="$MODEL" \
  CAIX_PREFILL_MITIGATION_PROMPT="$PROMPT" \
  CAIX_PREFILL_MITIGATION_OUTPUT="$OUT/run${run}.tokens" \
  CAIX_PREFILL_MITIGATION_META="$OUT/run${run}.meta" \
  CAIX_PREFILL_MITIGATION_MAX_TOKENS="$MAX_TOKENS" \
  swift test -c release \
    --filter PrefillMitigationExperimentTests/testRunMonolithicPrefillMitigationPrompt \
    > "$OUT/run${run}.log" 2>&1
done

first="$OUT/run1.tokens"
for run in $(seq 2 "$RUNS"); do
  cmp -s "$first" "$OUT/run${run}.tokens" || {
    echo "monolithic prefill determinism failed; token files differ under $OUT" >&2
    exit 1
  }
done

printf 'monolithic prefill deterministic across %s fresh processes: %s\n' \
  "$RUNS" "$(tr -d '\n' < "$first")"
