#!/usr/bin/env bash
# Self-test publication-gate option semantics without running the full gate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE="$SCRIPT_DIR/check-publication-gates.sh"

fail() {
  echo "error: $*" >&2
  exit 1
}

[[ -x "$GATE" ]] || fail "publication gate is not executable: $GATE"

help="$("$GATE" --help)"
grep -F -- '--strict-evidence requires tracked distributed evidence files.' <<<"$help" >/dev/null \
  || fail "help text must document --strict-evidence"
grep -F -- '--distributed checks the Thunderbolt readiness gate and implies --strict-evidence.' <<<"$help" >/dev/null \
  || fail "help text must document that --distributed implies --strict-evidence"
grep -F -- '--strict-benchmark-gaps fails when an eligible benchmark manifest row lacks raw evidence.' <<<"$help" >/dev/null \
  || fail "help text must document --strict-benchmark-gaps"

grep -F -- '--strict-evidence) STRICT_EVIDENCE=1; shift ;;' "$GATE" >/dev/null \
  || fail "argument parser must support --strict-evidence"
grep -F -- '--strict-benchmark-gaps) STRICT_BENCHMARK_GAPS=1; shift ;;' "$GATE" >/dev/null \
  || fail "argument parser must support --strict-benchmark-gaps"
grep -F -- 'check-distributed-evidence-contract.sh" --require-tracked' "$GATE" >/dev/null \
  || fail "strict evidence path must call check-distributed-evidence-contract.sh --require-tracked"
grep -F -- 'check-benchmark-gaps.sh" --strict' "$GATE" >/dev/null \
  || fail "strict benchmark gaps path must call check-benchmark-gaps.sh --strict"

awk '
  /\[\[ "\$RUN_DISTRIBUTED" == "1" \]\]/ { in_block = 1; found = 0 }
  in_block && /STRICT_EVIDENCE=1/ { found = 1 }
  in_block && /^fi$/ {
    if (found) {
      ok = 1
      exit 0
    }
    exit 1
  }
  END { if (!ok) exit 1 }
' "$GATE" || fail "--distributed must imply STRICT_EVIDENCE=1"

echo "publication gates contract ok"
