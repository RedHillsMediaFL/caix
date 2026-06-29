#!/usr/bin/env bash
# Check disk-pressure preflight pass/fail behavior without touching model payloads.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-disk-pressure-guard.sh

Exercises scripts/check-disk-pressure.sh against a temporary path.
Does not build, download, upload, benchmark, or touch models/exports.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
elif [[ "$#" -ne 0 ]]; then
  echo "unknown option: $1" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-disk-pressure.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

json="$("$SCRIPT_DIR/check-disk-pressure.sh" --path "$tmpdir" --floor-gib 0 --json)"
free_gib="$(JSON="$json" python3 - <<'PY'
import json
import os

doc = json.loads(os.environ["JSON"])
if doc.get("status") != "ok":
    raise SystemExit("expected ok status")
free = doc.get("free_gib")
if not isinstance(free, int) or free < 0:
    raise SystemExit("free_gib must be a non-negative integer")
print(free)
PY
)"

json_path="$tmpdir/json path \" check"
mkdir -p "$json_path"
json="$("$SCRIPT_DIR/check-disk-pressure.sh" --path "$json_path" --floor-gib 0 --json)"
JSON="$json" EXPECTED_PATH="$json_path" python3 - <<'PY'
import json
import os

doc = json.loads(os.environ["JSON"])
if doc.get("path") != os.environ["EXPECTED_PATH"]:
    raise SystemExit("json path was not preserved")
if doc.get("status") != "ok":
    raise SystemExit("expected ok status for quoted path")
PY

floor_gib=$((free_gib + 1))
if "$SCRIPT_DIR/check-disk-pressure.sh" --path "$tmpdir" --floor-gib "$floor_gib" --quiet; then
  echo "error: disk-pressure guard accepted a floor above available space" >&2
  exit 1
fi

if "$SCRIPT_DIR/check-disk-pressure.sh" --path "$tmpdir/missing" --quiet >/dev/null 2>&1; then
  echo "error: disk-pressure guard accepted a missing path" >&2
  exit 1
fi

mkdir -p "$tmpdir/bin"
cat > "$tmpdir/bin/df" <<'SH'
#!/usr/bin/env bash
printf 'Filesystem 1G-blocks Used Available Capacity Mounted on\n'
printf '/dev/test 100 1 not-a-number 1%% /tmp\n'
SH
chmod +x "$tmpdir/bin/df"
if PATH="$tmpdir/bin:$PATH" "$SCRIPT_DIR/check-disk-pressure.sh" --path "$tmpdir" --quiet >"$tmpdir/bad-df.txt" 2>&1; then
  echo "error: disk-pressure guard accepted malformed df output" >&2
  exit 1
fi
if ! grep -F "error: could not read numeric filesystem free space" "$tmpdir/bad-df.txt" >/dev/null; then
  echo "error: disk-pressure guard did not report malformed df output" >&2
  exit 1
fi

echo "disk pressure guard ok"
