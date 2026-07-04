#!/usr/bin/env bash
# Self-test dependency evidence collection/checking with local fixtures only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-dependency-evidence-contract.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

resolved="$tmpdir/Package.resolved"
pyproject="$tmpdir/pyproject.toml"
evidence="$tmpdir/DEPENDENCY_EVIDENCE.tsv"

cat > "$resolved" <<'JSON'
{
  "pins" : [
    {
      "identity" : "coreai-models",
      "kind" : "remoteSourceControl",
      "location" : "https://github.com/apple/coreai-models.git",
      "state" : {
        "branch" : "main",
        "revision" : "1111111111111111111111111111111111111111"
      }
    },
    {
      "identity" : "xgrammar",
      "kind" : "remoteSourceControl",
      "location" : "https://github.com/mlc-ai/xgrammar",
      "state" : {
        "revision" : "2222222222222222222222222222222222222222",
        "version" : "0.1.0"
      }
    }
  ],
  "version" : 3
}
JSON

cat > "$pyproject" <<'TOML'
[project]
dependencies = [
  "coreai-core==1.0.0b2",
  "coreai-torch==0.4.1",
  "coreai-opt==0.2.1",
]
TOML

"$SCRIPT_DIR/collect-dependency-evidence.sh" \
  --package-resolved "$resolved" \
  --coreai-pyproject "$pyproject" \
  --out "$evidence"

"$SCRIPT_DIR/check-dependency-evidence.sh" \
  --evidence "$evidence" \
  --package-resolved "$resolved" \
  --coreai-pyproject "$pyproject" >/dev/null

grep -F $'coreai-models\tswiftpm\tbranch\tmain\t1111111111111111111111111111111111111111' \
  "$evidence" >/dev/null \
  || { echo "error: fixture evidence did not record coreai-models branch revision" >&2; cat "$evidence" >&2; exit 1; }
grep -F $'xgrammar\tswiftpm\tversion\t0.1.0\t2222222222222222222222222222222222222222' \
  "$evidence" >/dev/null \
  || { echo "error: fixture evidence did not record xgrammar version revision" >&2; cat "$evidence" >&2; exit 1; }
grep -F $'coreai-torch\tpython\texact\t0.4.1' "$evidence" >/dev/null \
  || { echo "error: fixture evidence did not record exact coreai-torch dependency" >&2; cat "$evidence" >&2; exit 1; }

stale="$tmpdir/stale.tsv"
cp "$evidence" "$stale"
perl -0pi -e 's/2222222222222222222222222222222222222222/3333333333333333333333333333333333333333/' "$stale"
if "$SCRIPT_DIR/check-dependency-evidence.sh" \
    --evidence "$stale" \
    --package-resolved "$resolved" \
    --coreai-pyproject "$pyproject" >"$tmpdir/stale.out" 2>&1
then
  echo "error: stale dependency evidence unexpectedly passed" >&2
  exit 1
fi
grep -F 'is stale; regenerate with:' "$tmpdir/stale.out" >/dev/null \
  || { echo "error: stale evidence failure did not mention regenerate command" >&2; cat "$tmpdir/stale.out" >&2; exit 1; }

missing_pin="$tmpdir/missing-xgrammar.Package.resolved"
python3 - "$resolved" "$missing_pin" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
data = json.loads(source.read_text())
data["pins"] = [pin for pin in data["pins"] if pin["identity"] != "xgrammar"]
target.write_text(json.dumps(data, indent=2) + "\n")
PY
if "$SCRIPT_DIR/collect-dependency-evidence.sh" \
    --package-resolved "$missing_pin" \
    --coreai-pyproject "$pyproject" \
    --out "$tmpdir/missing.tsv" >"$tmpdir/missing.out" 2>&1
then
  echo "error: dependency evidence collection unexpectedly passed without xgrammar pin" >&2
  exit 1
fi
grep -F 'Package.resolved missing xgrammar' "$tmpdir/missing.out" >/dev/null \
  || { echo "error: missing-pin failure did not mention xgrammar" >&2; cat "$tmpdir/missing.out" >&2; exit 1; }

range_pyproject="$tmpdir/range.pyproject.toml"
cat > "$range_pyproject" <<'TOML'
[project]
dependencies = [
  "coreai-core==1.0.0b2",
  "coreai-torch>=0.4,<0.5",
  "coreai-opt==0.2.1",
]
TOML
if "$SCRIPT_DIR/collect-dependency-evidence.sh" \
    --package-resolved "$resolved" \
    --coreai-pyproject "$range_pyproject" \
    --out "$tmpdir/range.tsv" >"$tmpdir/range.out" 2>&1
then
  echo "error: dependency evidence collection unexpectedly passed with ranged coreai-torch" >&2
  exit 1
fi
grep -F 'missing exact dependency coreai-torch==...' "$tmpdir/range.out" >/dev/null \
  || { echo "error: ranged-python failure did not mention exact coreai-torch dependency" >&2; cat "$tmpdir/range.out" >&2; exit 1; }

echo "dependency evidence contract ok"
