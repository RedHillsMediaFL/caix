#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

fail() {
  echo "error: $*" >&2
  exit 1
}

swift_version="$(
  sed -nE 's/^[[:space:]]*static let version = "([^"]+)".*/\1/p' \
    Sources/PipelineCLI/BuildInfo.swift | head -1
)"
package_version="$(
  sed -nE 's/^VERSION="\$\{1:-([^}]*)\}".*/\1/p' scripts/package.sh | head -1
)"
formula_version="$(
  sed -nE 's/^[[:space:]]*version "([^"]+)".*/\1/p' Formula/caix.rb | head -1
)"

[[ -n "$swift_version" ]] || fail "could not read Sources/PipelineCLI/BuildInfo.swift version"
[[ -n "$package_version" ]] || fail "could not read scripts/package.sh default version"
[[ -n "$formula_version" ]] || fail "could not read Formula/caix.rb version"

"$REPO_DIR/scripts/check-release-version.sh" "$swift_version" >/dev/null

[[ "$swift_version" == "$package_version" ]] || {
  fail "package version $package_version does not match CLI version $swift_version"
}
[[ "$swift_version" == "$formula_version" ]] || {
  fail "formula version $formula_version does not match CLI version $swift_version"
}

echo "version sync ok: $swift_version"
