#!/usr/bin/env bash
# Emit exact dependency evidence needed for release, benchmark, and capability claims.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/collect-dependency-evidence.sh [--out <path>] [--package-resolved <path>] [--coreai-pyproject <path>]

Writes a TSV with the exact resolved Core AI dependency set. This is no-load: it only reads
Package.resolved and local checkout metadata.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT="-"
PACKAGE_RESOLVED="$REPO_DIR/Package.resolved"
COREAI_PYPROJECT="$REPO_DIR/.build/checkouts/coreai-models/python/pyproject.toml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="${2:?}"; shift 2 ;;
    --package-resolved) PACKAGE_RESOLVED="${2:?}"; shift 2 ;;
    --coreai-pyproject) COREAI_PYPROJECT="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$PACKAGE_RESOLVED" ]] || { echo "error: Package.resolved not found" >&2; exit 2; }
[[ -f "$COREAI_PYPROJECT" ]] || {
  echo "error: coreai-models python pyproject not found: $COREAI_PYPROJECT" >&2
  echo "hint: resolve COREAI_RUNTIME dependencies before collecting release evidence" >&2
  exit 2
}

tmp="$(mktemp "${TMPDIR:-/tmp}/caix-dependency-evidence.XXXXXX")"
trap 'rm -f "$tmp"' EXIT

python3 - "$PACKAGE_RESOLVED" "$COREAI_PYPROJECT" "$REPO_DIR" > "$tmp" <<'PY'
import json
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError as exc:
    raise SystemExit("error: Python 3.11+ tomllib is required") from exc

resolved_path = Path(sys.argv[1])
pyproject_path = Path(sys.argv[2])
repo_dir = Path(sys.argv[3])

resolved = json.loads(resolved_path.read_text())
pins = {pin["identity"]: pin for pin in resolved.get("pins", [])}

def rel(path: Path) -> str:
    try:
        return str(path.relative_to(repo_dir))
    except ValueError:
        return str(path)

def swift_pin(identity: str, component: str, note: str) -> list[str]:
    try:
        pin = pins[identity]
    except KeyError as exc:
        raise SystemExit(f"error: Package.resolved missing {identity}") from exc
    state = pin.get("state", {})
    revision = state.get("revision")
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit(f"error: {identity} revision is not a 40-character SHA")
    version = state.get("version") or state.get("branch") or ""
    version_kind = "version" if state.get("version") else ("branch" if state.get("branch") else "revision")
    return [
        component,
        "swiftpm",
        version_kind,
        version,
        revision,
        pin.get("location", ""),
        rel(resolved_path),
        note,
    ]

def python_dep(name: str, note: str) -> list[str]:
    data = tomllib.loads(pyproject_path.read_text())
    deps = data.get("project", {}).get("dependencies", [])
    pattern = re.compile(rf"^{re.escape(name)}==([^;\s]+)$")
    for dep in deps:
        match = pattern.match(dep)
        if match:
            return [
                name,
                "python",
                "exact",
                match.group(1),
                "",
                "https://pypi.org/project/" + name + "/",
                rel(pyproject_path),
                note,
            ]
    raise SystemExit(f"error: {pyproject_path} missing exact dependency {name}==...")

rows = [
    [
        "component",
        "ecosystem",
        "version_kind",
        "version_or_branch",
        "revision",
        "location",
        "evidence_source",
        "notes",
    ],
    swift_pin(
        "coreai-models",
        "coreai-models",
        "CoreAILM runtime package; development tracks main, release evidence records this SHA.",
    ),
    swift_pin(
        "xgrammar",
        "xgrammar",
        "Constrained decoding dependency in the caix Swift build graph.",
    ),
    python_dep(
        "coreai-torch",
        "Python export/conversion dependency declared by the resolved coreai-models checkout.",
    ),
    python_dep(
        "coreai-core",
        "Python Core AI base dependency declared by the resolved coreai-models checkout.",
    ),
    python_dep(
        "coreai-opt",
        "Python Core AI optimization dependency declared by the resolved coreai-models checkout.",
    ),
]

for row in rows:
    print("\t".join(row))
PY

if [[ "$OUT" == "-" ]]; then
  cat "$tmp"
else
  mkdir -p "$(dirname "$OUT")"
  cp "$tmp" "$OUT"
fi
