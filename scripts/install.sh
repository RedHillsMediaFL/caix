#!/usr/bin/env bash
# caix installer (BETA) — builds the release binary. Requires a Core AI–capable Xcode/Swift
# toolchain (see README: "Requirements"). Dependencies are fetched automatically by SwiftPM.
set -euo pipefail

REPO_URL="${CAIX_REPO_URL:-https://github.com/RedHillsMediaFL/caix.git}"
INSTALL_DIR="${CAIX_DIR:-$HOME/caix}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" 2>/dev/null && pwd || pwd)"
if [ -f "$script_dir/../Package.swift" ] && [ -d "$script_dir/../Sources" ]; then
  DIR="$(cd "$script_dir/.." && pwd)"
else
  if ! command -v git >/dev/null 2>&1; then
    echo "  ✗ 'git' not found. Install Xcode command line tools or Git, then re-run the installer."
    exit 1
  fi
  if [ -d "$INSTALL_DIR/.git" ]; then
    echo "  updating $INSTALL_DIR…"
    git -C "$INSTALL_DIR" pull --ff-only
  elif [ -e "$INSTALL_DIR" ]; then
    echo "  ✗ $INSTALL_DIR exists but is not a git checkout."
    echo "    Set CAIX_DIR=/path/to/empty-dir or move the existing path."
    exit 1
  else
    echo "  cloning $REPO_URL → $INSTALL_DIR…"
    git clone "$REPO_URL" "$INSTALL_DIR"
  fi
  DIR="$INSTALL_DIR"
fi

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

mkdir -p "$HOME/.local/bin"
if ln -sf "$DIR/caix" "$HOME/.local/bin/caix" 2>/dev/null; then
  SHIM="$HOME/.local/bin/caix"
else
  SHIM="$DIR/caix"
fi

echo ""
echo "✓ Built: $DIR/.build/release/caix"
echo "✓ Launcher: $SHIM"
echo ""
echo "Next steps:"
echo "  1) Download a model (see README → 'Get a model'), e.g. into $DIR/models/exports/"
echo "  2) caix serve    # or: $DIR/caix serve"
echo "  3) open http://localhost:1237   (dashboard)  ·  /chat  (chat)"
