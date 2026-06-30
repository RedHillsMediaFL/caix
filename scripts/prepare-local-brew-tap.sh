#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/prepare-local-brew-tap.sh --tarball <caix-version-macos-arm64.tar.gz> [options]

Writes a local Homebrew tap formula for a versioned caix tarball.

Options:
  --tarball <path>       release tarball built by scripts/package.sh
  --formula <path>       source formula (default: Formula/caix.rb beside checkout or dist)
  --tap <owner/name>     local tap name (default: RedHillsMediaFL/caix)
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$SCRIPT_DIR/Formula/caix.rb" ]]; then
  default_formula="$SCRIPT_DIR/Formula/caix.rb"
else
  default_formula="$(cd "$SCRIPT_DIR/.." && pwd)/Formula/caix.rb"
fi

tarball=""
formula="$default_formula"
tap="RedHillsMediaFL/caix"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tarball) tarball="${2:?}"; shift 2 ;;
    --formula) formula="${2:?}"; shift 2 ;;
    --tap) tap="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) echo "error: unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$tarball" ]] || { usage >&2; exit 2; }
command -v brew >/dev/null 2>&1 || die "brew not found"
command -v shasum >/dev/null 2>&1 || die "shasum not found"
[[ -f "$formula" ]] || die "formula not found: $formula"
[[ -f "$tarball" ]] || die "tarball not found: $tarball"

tarball_path="$(python3 - "$tarball" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
tarball_url="$(python3 - "$tarball_path" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).as_uri())
PY
)"
sha256="$(shasum -a 256 "$tarball_path" | awk '{print $1}')"

tap_lc="$(printf '%s\n' "$tap" | tr '[:upper:]' '[:lower:]')"
if ! brew tap 2>/dev/null | grep -qi "^${tap_lc}$"; then
  brew tap-new "$tap"
fi
tap_dir="$(brew --repository "$tap")"
mkdir -p "$tap_dir/Formula"
out_formula="$tap_dir/Formula/caix.rb"

python3 - "$formula" "$out_formula" "$tarball_url" "$sha256" <<'PY'
from pathlib import Path
import re
import sys

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
url = sys.argv[3]
sha256 = sys.argv[4]

lines = source.read_text(encoding="utf-8").splitlines(keepends=True)
out = []
inserted = False
for line in lines:
    stripped = line.lstrip()
    if stripped.startswith("url ") or stripped.startswith("sha256 "):
        continue
    if "Head-only until the tap has a tested 0.x release tarball" in line:
        continue
    out.append(line)
    if not inserted and re.match(r"\s*version\s+\"", line):
        indent = line[: len(line) - len(stripped)]
        out.append(f'{indent}url "{url}"\n')
        out.append(f'{indent}sha256 "{sha256}"\n')
        inserted = True

if not inserted:
    raise SystemExit("could not find version line in formula")

destination.write_text("".join(out), encoding="utf-8")
PY

echo "local tap formula wrote: $out_formula"
echo "tarball: $tarball_path"
echo "sha256: $sha256"
echo "install: brew install $tap_lc/caix"
