#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

BUILD_INFO="$REPO_DIR/Sources/PipelineCLI/BuildInfo.swift"
PACKAGE_SCRIPT="$REPO_DIR/scripts/package.sh"
FORMULA="$REPO_DIR/Formula/caix.rb"

usage() {
  cat <<'USAGE'
Usage: scripts/check-version-sync.sh [--build-info <path>] [--package-script <path>] [--formula <path>]

Checks that CLI, package, and formula versions match and satisfy release-version policy.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-info) BUILD_INFO="${2:?}"; shift 2 ;;
    --package-script) PACKAGE_SCRIPT="${2:?}"; shift 2 ;;
    --formula) FORMULA="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) echo "error: unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

fail() {
  echo "error: $*" >&2
  exit 1
}

swift_version="$(
  sed -nE 's/^[[:space:]]*static let version = "([^"]+)".*/\1/p' \
    "$BUILD_INFO" | head -1
)"
package_version="$(
  sed -nE 's/^VERSION="\$\{1:-([^}]*)\}".*/\1/p' "$PACKAGE_SCRIPT" | head -1
)"
formula_version="$(
  sed -nE 's/^[[:space:]]*version "([^"]+)".*/\1/p' "$FORMULA" | head -1
)"

[[ -n "$swift_version" ]] || fail "could not read BuildInfo version from $BUILD_INFO"
[[ -n "$package_version" ]] || fail "could not read package default version from $PACKAGE_SCRIPT"
[[ -n "$formula_version" ]] || fail "could not read formula version from $FORMULA"

"$REPO_DIR/scripts/check-release-version.sh" "$swift_version" >/dev/null

[[ "$swift_version" == "$package_version" ]] || {
  fail "package version $package_version does not match CLI version $swift_version"
}
[[ "$swift_version" == "$formula_version" ]] || {
  fail "formula version $formula_version does not match CLI version $swift_version"
}

echo "version sync ok: $swift_version"
