#!/usr/bin/env bash
# Check host requirements for a Core AI-linked caix build.
set -euo pipefail

NO_FAIL=0
QUIET=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-fail) NO_FAIL=1 ;;
    --quiet) QUIET=1 ;;
    -h|--help)
      cat <<'USAGE'
Usage: scripts/check-coreai-runtime.sh [--no-fail] [--quiet]

Checks Apple silicon, macOS 27+, CoreAI.framework, and Swift.
USAGE
      exit 0
      ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

failures=0

check() {
  local required="$1" name="$2" detail="$3"
  shift 3
  if "$@"; then
    [ "$QUIET" -eq 1 ] || printf '[ok] %s: %s\n' "$name" "$detail"
  elif [ "$required" = "required" ]; then
    failures=$((failures + 1))
    [ "$QUIET" -eq 1 ] || printf '[fail] %s: %s\n' "$name" "$detail"
  else
    [ "$QUIET" -eq 1 ] || printf '[warn] %s: %s\n' "$name" "$detail"
  fi
}

arch="$(uname -m)"
macos="$(sw_vers -productVersion 2>/dev/null || echo 0.0.0)"
major="${macos%%.*}"
case "$major" in
  ''|*[!0-9]*) major=0 ;;
esac

coreai_path=""
for p in \
  /System/Library/Frameworks/CoreAI.framework \
  /System/Library/PrivateFrameworks/CoreAI.framework
do
  if [ -d "$p" ]; then
    coreai_path="$p"
    break
  fi
done

swift_path="$(command -v swift 2>/dev/null || true)"

check required "Apple silicon" "$arch" test "$arch" = "arm64"
check required "macOS 27+" "$macos" test "$major" -ge 27
check required "CoreAI.framework" "${coreai_path:-not found}" test -n "$coreai_path"
check optional "Swift toolchain" "${swift_path:-not found}" test -n "$swift_path"

if [ "$failures" -gt 0 ] && [ "$NO_FAIL" -eq 0 ]; then
  exit 1
fi
