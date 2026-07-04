#!/usr/bin/env bash
# Self-test cross-file version synchronization with local fixtures only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-version-sync-contract.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

build_info="$tmpdir/BuildInfo.swift"
package_script="$tmpdir/package.sh"
formula="$tmpdir/caix.rb"

write_fixture() {
  local cli="$1"
  local pkg="$2"
  local formula_version="$3"

  cat > "$build_info" <<EOF
enum CaixBuildInfo {
    static let version = "$cli"
}
EOF

  cat > "$package_script" <<EOF
#!/usr/bin/env bash
VERSION="\${1:-$pkg}"
EOF

  cat > "$formula" <<EOF
class Caix < Formula
  version "$formula_version"
end
EOF
}

run_sync() {
  "$SCRIPT_DIR/check-version-sync.sh" \
    --build-info "$build_info" \
    --package-script "$package_script" \
    --formula "$formula"
}

write_fixture 0.2.11-beta 0.2.11-beta 0.2.11-beta
run_sync >/dev/null

write_fixture 0.2.11-beta 0.2.12-beta 0.2.11-beta
if run_sync >"$tmpdir/package.out" 2>&1; then
  echo "error: version sync unexpectedly passed with mismatched package version" >&2
  exit 1
fi
grep -F 'package version 0.2.12-beta does not match CLI version 0.2.11-beta' \
  "$tmpdir/package.out" >/dev/null \
  || { echo "error: package mismatch failure changed unexpectedly" >&2; cat "$tmpdir/package.out" >&2; exit 1; }

write_fixture 0.2.11-beta 0.2.11-beta 0.2.12-beta
if run_sync >"$tmpdir/formula.out" 2>&1; then
  echo "error: version sync unexpectedly passed with mismatched formula version" >&2
  exit 1
fi
grep -F 'formula version 0.2.12-beta does not match CLI version 0.2.11-beta' \
  "$tmpdir/formula.out" >/dev/null \
  || { echo "error: formula mismatch failure changed unexpectedly" >&2; cat "$tmpdir/formula.out" >&2; exit 1; }

cat > "$build_info" <<'EOF'
enum CaixBuildInfo {
}
EOF
write_fixture_missing_pkg="$tmpdir/package.sh"
cat > "$write_fixture_missing_pkg" <<'EOF'
#!/usr/bin/env bash
VERSION="${1:-0.2.11-beta}"
EOF
cat > "$formula" <<'EOF'
class Caix < Formula
  version "0.2.11-beta"
end
EOF
if "$SCRIPT_DIR/check-version-sync.sh" \
    --build-info "$build_info" \
    --package-script "$write_fixture_missing_pkg" \
    --formula "$formula" >"$tmpdir/missing.out" 2>&1
then
  echo "error: version sync unexpectedly passed without BuildInfo version" >&2
  exit 1
fi
grep -F 'could not read BuildInfo version' "$tmpdir/missing.out" >/dev/null \
  || { echo "error: missing BuildInfo failure changed unexpectedly" >&2; cat "$tmpdir/missing.out" >&2; exit 1; }

write_fixture 1.0.0 1.0.0 1.0.0
if run_sync >"$tmpdir/beta.out" 2>&1; then
  echo "error: version sync unexpectedly passed a 1.0.0 release while Core AI beta gate is active" >&2
  exit 1
fi
grep -F 'Core AI is beta; keep caix releases below 1.0.0' "$tmpdir/beta.out" >/dev/null \
  || { echo "error: beta gate failure did not propagate through version sync" >&2; cat "$tmpdir/beta.out" >&2; exit 1; }

echo "version sync contract ok"
