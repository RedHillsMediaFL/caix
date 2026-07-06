#!/usr/bin/env bash
# Self-test release package script invariants without building or packaging.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE="$SCRIPT_DIR/package.sh"

fail() {
  echo "error: $*" >&2
  exit 1
}

[[ -x "$PACKAGE" ]] || fail "package script is not executable: $PACKAGE"

grep -F -- 'DEFAULT_TMPDIR="$DIR/.tmp/coreai-tmp"' "$PACKAGE" >/dev/null \
  || fail "package script must define an SSD-backed default TMPDIR"
grep -F -- 'is_system_tmpdir()' "$PACKAGE" >/dev/null \
  || fail "package script must detect macOS system TMPDIR locations"
grep -F -- '/var/folders/*|/private/var/folders/*' "$PACKAGE" >/dev/null \
  || fail "package script must treat macOS /var/folders TMPDIR as unsafe"
grep -F -- 'export TMPDIR="$DEFAULT_TMPDIR"' "$PACKAGE" >/dev/null \
  || fail "package script must export the SSD-backed default TMPDIR when TMPDIR is unset or system-backed"
grep -F -- 'mkdir -p "${TMPDIR%/}"' "$PACKAGE" >/dev/null \
  || fail "package script must create the effective TMPDIR before staging"
grep -F -- 'mktemp -d "${TMPDIR%/}/caix-package.XXXXXX"' "$PACKAGE" >/dev/null \
  || fail "package staging must allocate under TMPDIR"
grep -F -- 'trap cleanup EXIT' "$PACKAGE" >/dev/null \
  || fail "package staging directory must be cleaned up on exit"
grep -F -- 'shasum -a 256 "$DIR/dist/$NAME.tar.gz" > "$DIR/dist/$NAME.tar.gz.sha256"' "$PACKAGE" >/dev/null \
  || fail "package script must write a SHA-256 sidecar for the tarball"
grep -F -- 'echo "  sha256: $SHA256"' "$PACKAGE" >/dev/null \
  || fail "package script must print the release tarball SHA-256"

echo "package contract ok"
