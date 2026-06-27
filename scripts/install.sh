#!/usr/bin/env bash
# caix installer (BETA) — builds the release binary. Requires a Core AI–capable Xcode/Swift
# toolchain (see README: "Requirements"). Dependencies are fetched automatically by SwiftPM.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "→ caix installer (beta)"
echo "  checking toolchain…"
if ! command -v swift >/dev/null 2>&1; then
  echo "  ✗ 'swift' not found. Install Xcode (beta) from the App Store / developer.apple.com, then:"
  echo "      sudo xcode-select -s /Applications/Xcode-beta.app   # or your Core AI–capable Xcode"
  exit 1
fi
swift --version | sed 's/^/    /'

echo "  building (first build fetches deps + compiles — a few minutes)…"
( cd "$DIR" && COREAI_RUNTIME=1 swift build -c release )

echo ""
echo "✓ Built: $DIR/.build/release/caix"
echo ""
echo "Next steps:"
echo "  1) Download a model (see README → 'Get a model'), e.g. into $DIR/models/exports/"
echo "  2) ./caix serve"
echo "  3) open http://localhost:1237   (dashboard)  ·  /chat  (chat)"
