#!/usr/bin/env bash
# Build caix and produce a self-contained release tarball: a prebuilt binary + the web UI +
# converter + launcher, so users can download, extract, and `./caix serve` with NO build step
# (on a Mac with a matching Core AI runtime). Usage: scripts/package.sh [version]
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-0.2.8-beta}"
ARCH="$(uname -m)"
NAME="caix-${VERSION}-macos-${ARCH}"

cd "$DIR"
"$DIR/scripts/check-release-version.sh" "$VERSION"
"$DIR/scripts/check-coreai-runtime.sh"
echo "→ building release binary…"
COREAI_RUNTIME=1 swift build -c release

BUILT_VERSION="$(.build/release/caix --version | awk '{print $2}')"
if [ "$BUILT_VERSION" != "$VERSION" ]; then
  echo "✗ version mismatch: binary reports $BUILT_VERSION, package requested $VERSION" >&2
  echo "  update Sources/PipelineCLI/BuildInfo.swift or pass the matching version." >&2
  exit 1
fi
"$DIR/scripts/check-brew-distributed.sh" \
  --caix "$DIR/.build/release/caix" \
  --ready \
  --manifest "$DIR/docs/examples/cluster-stage-manifest.json"

STAGE="$(mktemp -d)/$NAME"
mkdir -p "$STAGE/bin" "$STAGE/models/exports" "$STAGE/scripts" "$STAGE/docs/examples" "$STAGE/Formula"
cp .build/release/caix          "$STAGE/bin/caix"
cp caix README.md LICENSE       "$STAGE/" 2>/dev/null || true
cp Formula/caix.rb              "$STAGE/Formula/"
cp -R web python                "$STAGE/"
cp models/registry.json         "$STAGE/models/"
cp docs/examples/cluster-stage-manifest.json "$STAGE/docs/examples/"
cp scripts/install.sh scripts/check-coreai-runtime.sh scripts/check-brew-distributed.sh \
  scripts/check-distributed-readiness.sh scripts/check-stage-bundle-copy.sh \
  scripts/check-tiny-cluster-smoke.sh scripts/prepare-local-brew-tap.sh "$STAGE/scripts/" 2>/dev/null || true
cp -R scripts/lib "$STAGE/scripts/" 2>/dev/null || true
chmod +x "$STAGE/caix" "$STAGE/bin/caix"
printf "%s\n" "$VERSION" > "$STAGE/VERSION"

mkdir -p "$DIR/dist"
( cd "$(dirname "$STAGE")" && tar -czf "$DIR/dist/$NAME.tar.gz" "$NAME" )
mkdir -p "$DIR/dist/Formula"
cp scripts/prepare-local-brew-tap.sh "$DIR/dist/"
cp Formula/caix.rb "$DIR/dist/Formula/"
SIZE="$(du -h "$DIR/dist/$NAME.tar.gz" | awk '{print $1}')"
echo "✓ dist/$NAME.tar.gz  ($SIZE)"
echo "  contains a prebuilt binary — recipients run ./caix serve with no build."
echo "  local tap helper: dist/prepare-local-brew-tap.sh"
