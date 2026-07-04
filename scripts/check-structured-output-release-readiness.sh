#!/usr/bin/env bash
# No-load readiness check before lifting public structured-output claims.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-structured-output-release-readiness.sh --evidence <smoke.json> [--dependency-evidence <tsv>] [--allow-branch-dependencies]

Validates captured structured-output smoke evidence and dependency provenance before a release
review decides whether to lift the public response_format copy gate.

Strict mode rejects branch-based structured-output dependencies such as xgrammar on main.
Use --allow-branch-dependencies only for development reporting; do not use it to justify public copy.
This check does not build, load models, run inference, contact the network, or publish claims.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEPENDENCY_EVIDENCE="$REPO_DIR/benchmarks/DEPENDENCY_EVIDENCE.tsv"
ALLOW_BRANCH_DEPS=0
evidence_files=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --evidence) evidence_files+=("${2:?}"); shift 2 ;;
    --dependency-evidence) DEPENDENCY_EVIDENCE="${2:?}"; shift 2 ;;
    --allow-branch-dependencies) ALLOW_BRANCH_DEPS=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) echo "error: unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${#evidence_files[@]}" -eq 0 ]]; then
  echo "error: provide at least one --evidence <smoke.json>" >&2
  usage >&2
  exit 2
fi

"$SCRIPT_DIR/check-structured-output-evidence.sh" "${evidence_files[@]}" >/dev/null

python3 - "${evidence_files[@]}" <<'PY'
import json
import sys
from pathlib import Path

complete_stops = {"eos", "stopSequence"}

for raw_path in sys.argv[1:]:
    path = Path(raw_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    stop_reason = payload.get("stop_reason")
    if stop_reason not in complete_stops:
        print(
            f"error: {path}: structured-output release evidence must stop by eos or stopSequence, "
            f"not `{stop_reason}`",
            file=sys.stderr,
        )
        raise SystemExit(1)

    tps = float(payload.get("decode_tokens_per_second", 0))
    if tps <= 0:
        print(
            f"error: {path}: structured-output release evidence must record positive decode_tokens_per_second",
            file=sys.stderr,
        )
        raise SystemExit(1)
PY

[[ -f "$DEPENDENCY_EVIDENCE" ]] || {
  echo "error: dependency evidence file not found: $DEPENDENCY_EVIDENCE" >&2
  exit 1
}

python3 - "$DEPENDENCY_EVIDENCE" "$ALLOW_BRANCH_DEPS" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
allow_branch = sys.argv[2] == "1"

with path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle, delimiter="\t")
    rows = list(reader)

expected_header = [
    "component",
    "ecosystem",
    "version_kind",
    "version_or_branch",
    "revision",
    "location",
    "evidence_source",
    "notes",
]
if reader.fieldnames != expected_header:
    print(f"error: unexpected dependency evidence header in {path}", file=sys.stderr)
    raise SystemExit(1)

by_component = {row["component"]: row for row in rows}
missing = [name for name in ["coreai-models", "xgrammar"] if name not in by_component]
if missing:
    print(f"error: missing structured-output dependency evidence row(s): {', '.join(missing)}", file=sys.stderr)
    raise SystemExit(1)

for name in ["coreai-models", "xgrammar"]:
    row = by_component[name]
    revision = row.get("revision", "")
    if len(revision) != 40 or any(ch not in "0123456789abcdef" for ch in revision.lower()):
        print(f"error: {name} must record a 40-character resolved revision", file=sys.stderr)
        raise SystemExit(1)
    if not row.get("location"):
        print(f"error: {name} must record a dependency location", file=sys.stderr)
        raise SystemExit(1)

xgrammar = by_component["xgrammar"]
if xgrammar.get("version_kind") == "branch" and not allow_branch:
    branch = xgrammar.get("version_or_branch", "")
    revision = xgrammar.get("revision", "")
    print(
        "error: xgrammar is still a branch dependency "
        f"({branch}@{revision}); pin or vendor it before lifting public structured-output claims",
        file=sys.stderr,
    )
    raise SystemExit(1)

if xgrammar.get("version_kind") == "branch":
    print(
        "warning: xgrammar remains branch-based; development evidence only, not public release readiness",
        file=sys.stderr,
    )

print("structured-output dependency evidence ok")
PY

echo "structured-output release readiness ok"
