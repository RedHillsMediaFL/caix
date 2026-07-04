#!/usr/bin/env bash
# Self-test release-version policy with local fixtures only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-release-version-contract.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

"$SCRIPT_DIR/check-release-version.sh" 0.2.11-beta >/dev/null
"$SCRIPT_DIR/check-release-version.sh" v0.2.11-beta >/dev/null
"$SCRIPT_DIR/check-release-version.sh" --dev-ok 0.2.12-dev >/dev/null
"$SCRIPT_DIR/check-release-version.sh" --coreai-stable 1.0.0 >/dev/null

if "$SCRIPT_DIR/check-release-version.sh" 1.0.0 >"$tmpdir/major.out" 2>&1; then
  echo "error: 1.0.0 unexpectedly passed while Core AI beta gate is active" >&2
  exit 1
fi
grep -F 'Core AI is beta; keep caix releases below 1.0.0' "$tmpdir/major.out" >/dev/null \
  || { echo "error: major-version failure did not mention Core AI beta gate" >&2; cat "$tmpdir/major.out" >&2; exit 1; }

if "$SCRIPT_DIR/check-release-version.sh" 0.2.12-dev >"$tmpdir/dev.out" 2>&1; then
  echo "error: dev release version unexpectedly passed without --dev-ok" >&2
  exit 1
fi
grep -F 'release version must not contain dev' "$tmpdir/dev.out" >/dev/null \
  || { echo "error: dev-version failure did not mention dev" >&2; cat "$tmpdir/dev.out" >&2; exit 1; }

if "$SCRIPT_DIR/check-release-version.sh" 0.2 >"$tmpdir/invalid.out" 2>&1; then
  echo "error: malformed release version unexpectedly passed" >&2
  exit 1
fi
grep -F 'version must look like v0.2.0-beta or 0.2.0' "$tmpdir/invalid.out" >/dev/null \
  || { echo "error: malformed-version failure did not mention expected shape" >&2; cat "$tmpdir/invalid.out" >&2; exit 1; }

if "$SCRIPT_DIR/check-release-version.sh" 0.2.11 0.2.12 >"$tmpdir/extra.out" 2>&1; then
  echo "error: multiple release versions unexpectedly passed" >&2
  exit 1
fi
grep -F 'only one version is accepted' "$tmpdir/extra.out" >/dev/null \
  || { echo "error: multiple-version failure did not mention single version" >&2; cat "$tmpdir/extra.out" >&2; exit 1; }

echo "release version contract ok"
