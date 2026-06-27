#!/usr/bin/env bash
# Build caix and produce a self-contained release tarball: a prebuilt binary + the web UI +
# converter + launcher, so users can download, extract, and `./caix serve` with NO build step
# (on a Mac with a matching Core AI runtime). Usage: scripts/package.sh [version]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-0.1.0-beta}"
ARCH="$(uname -m)"
NAME="caix-${VERSION}-macos-${ARCH}"

cd "$DIR"
echo "→ building release binary…"
COREAI_RUNTIME=1 swift build -c release

STAGE="$(mktemp -d)/$NAME"
mkdir -p "$STAGE/bin" "$STAGE/models/exports" "$STAGE/scripts"
cp .build/release/caix          "$STAGE/bin/caix"
cp caix README.md LICENSE       "$STAGE/" 2>/dev/null || true
cp -R web python                "$STAGE/"
cp models/registry.json         "$STAGE/models/"
cp scripts/install.sh           "$STAGE/scripts/" 2>/dev/null || true
chmod +x "$STAGE/caix" "$STAGE/bin/caix"
printf "%s\n" "$VERSION" > "$STAGE/VERSION"

mkdir -p "$DIR/dist"
( cd "$(dirname "$STAGE")" && tar -czf "$DIR/dist/$NAME.tar.gz" "$NAME" )
SIZE="$(du -h "$DIR/dist/$NAME.tar.gz" | awk '{print $1}')"
echo "✓ dist/$NAME.tar.gz  ($SIZE)"
echo "  contains a prebuilt binary — recipients run ./caix serve with no build."
