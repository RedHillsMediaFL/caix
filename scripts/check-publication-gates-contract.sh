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
grep -F -- 'Requires a local benchmarks/revisions.tsv; run scripts/collect-model-revisions.sh first.' <<<"$help" >/dev/null \
  || fail "help text must document the required local benchmark revisions artifact"
grep -F -- 'Uses .tmp/coreai-tmp when TMPDIR is unset or points at macOS system temp.' <<<"$help" >/dev/null \
  || fail "help text must document SSD-backed TMPDIR behavior"

grep -F -- '--strict-evidence) STRICT_EVIDENCE=1; shift ;;' "$GATE" >/dev/null \
  || fail "argument parser must support --strict-evidence"
grep -F -- '--strict-benchmark-gaps) STRICT_BENCHMARK_GAPS=1; shift ;;' "$GATE" >/dev/null \
  || fail "argument parser must support --strict-benchmark-gaps"
grep -F -- 'check-distributed-evidence-contract.sh" --require-tracked' "$GATE" >/dev/null \
  || fail "strict evidence path must call check-distributed-evidence-contract.sh --require-tracked"
grep -F -- 'check-benchmark-gaps.sh" --strict' "$GATE" >/dev/null \
  || fail "strict benchmark gaps path must call check-benchmark-gaps.sh --strict"
grep -F -- 'check-package-contract.sh' "$GATE" >/dev/null \
  || fail "publication gate must run package contract"
grep -F -- 'check-model-revisions-contract.sh' "$GATE" >/dev/null \
  || fail "publication gate must run model revisions contract"
grep -F -- 'check-model-revisions.sh' "$GATE" >/dev/null \
  || fail "publication gate must validate benchmark model revisions"
grep -F -- 'check-converter-temp-contract.sh' "$GATE" >/dev/null \
  || fail "publication gate must run converter temp contract"
grep -F -- 'check-gguf-dequant-contract.sh' "$GATE" >/dev/null \
  || fail "publication gate must run GGUF dequant contract"
grep -F -- 'DEFAULT_TMPDIR="$REPO_DIR/.tmp/coreai-tmp"' "$GATE" >/dev/null \
  || fail "publication gate must define an SSD-backed default TMPDIR"
grep -F -- 'is_system_tmpdir()' "$GATE" >/dev/null \
  || fail "publication gate must detect macOS system TMPDIR locations"
grep -F -- '/var/folders/*|/private/var/folders/*' "$GATE" >/dev/null \
  || fail "publication gate must treat macOS /var/folders TMPDIR as unsafe"
grep -F -- 'export TMPDIR="$DEFAULT_TMPDIR"' "$GATE" >/dev/null \
  || fail "publication gate must export the SSD-backed default TMPDIR when TMPDIR is unset or system-backed"
grep -F -- 'mkdir -p "${TMPDIR%/}"' "$GATE" >/dev/null \
  || fail "publication gate must create the effective TMPDIR before running checks"

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
