#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TEAM_ID="${TEAM_ID:-34UZ7Y2KSW}"
BUNDLE_ID="${BUNDLE_ID:-com.redhillsmediafl.CaixChat}"
ASC_KEY_PATH="${ASC_KEY_PATH:-}"

if [[ -z "$ASC_KEY_PATH" ]]; then
  echo "Set ASC_KEY_PATH to your App Store Connect AuthKey_*.p8 file" >&2
  exit 2
fi
if [[ ! -f "$ASC_KEY_PATH" ]]; then
  echo "Missing App Store Connect key at ASC_KEY_PATH=$ASC_KEY_PATH" >&2
  exit 2
fi

ASC_KEY_ID="${ASC_KEY_ID:-$(basename "$ASC_KEY_PATH" | sed -E 's/AuthKey_([A-Z0-9]+)\.p8/\1/')}"
ASC_ISSUER_ID="${ASC_ISSUER_ID:-}"
INTERNAL_TESTER_EMAIL="${INTERNAL_TESTER_EMAIL:-}"

if [[ -z "$ASC_KEY_ID" ]]; then
  echo "Set ASC_KEY_ID or use an ASC_KEY_PATH named AuthKey_<KEYID>.p8" >&2
  exit 2
fi
if [[ -z "$ASC_ISSUER_ID" ]]; then
  echo "Set ASC_ISSUER_ID to the App Store Connect issuer UUID" >&2
  exit 2
fi
if [[ -z "$INTERNAL_TESTER_EMAIL" ]]; then
  echo "Set INTERNAL_TESTER_EMAIL to the internal TestFlight tester email" >&2
  exit 2
fi

if ! command -v fastlane >/dev/null 2>&1; then
  echo "Missing fastlane. Install it with: brew install fastlane" >&2
  exit 4
fi

FASTLANE_SKIP_UPDATE_CHECK=1 \
ASC_KEY_PATH="$ASC_KEY_PATH" \
ASC_KEY_ID="$ASC_KEY_ID" \
ASC_ISSUER_ID="$ASC_ISSUER_ID" \
TEAM_ID="$TEAM_ID" \
BUNDLE_ID="$BUNDLE_ID" \
fastlane ios ensure_app_record

APP_COUNT="$(
  xcrun altool --list-apps \
    --filter-bundle-id "$BUNDLE_ID" \
    --api-key "$ASC_KEY_ID" \
    --api-issuer "$ASC_ISSUER_ID" \
    --p8-file-path "$ASC_KEY_PATH" \
    --output-format json 2>/dev/null \
  | python3 -c 'import json,sys; print(len(json.load(sys.stdin)))'
)"

if [[ "$APP_COUNT" == "0" ]]; then
  cat >&2 <<EOF
Fastlane did not find or create the App Store Connect app record for $BUNDLE_ID.

Expected record:
  Name: CAIX Chat
  Bundle ID: $BUNDLE_ID
  SKU: caix-chat-ios-0001
  Primary language: English (U.S.)
EOF
  exit 3
fi

rm -rf build/CaixChat.xcarchive build/export

xcodebuild archive \
  -project CaixChat.xcodeproj \
  -scheme CaixChat \
  -archivePath build/CaixChat.xcarchive \
  -destination 'generic/platform=iOS' \
  -allowProvisioningUpdates \
  -authenticationKeyPath "$ASC_KEY_PATH" \
  -authenticationKeyID "$ASC_KEY_ID" \
  -authenticationKeyIssuerID "$ASC_ISSUER_ID" \
  CODE_SIGN_STYLE=Automatic \
  DEVELOPMENT_TEAM="$TEAM_ID"

xcodebuild -exportArchive \
  -archivePath build/CaixChat.xcarchive \
  -exportOptionsPlist ExportOptions.plist \
  -exportPath build/export \
  -allowProvisioningUpdates \
  -authenticationKeyPath "$ASC_KEY_PATH" \
  -authenticationKeyID "$ASC_KEY_ID" \
  -authenticationKeyIssuerID "$ASC_ISSUER_ID"

APP_VERSION="$(
  plutil -extract ApplicationProperties.CFBundleShortVersionString raw \
    -o - build/CaixChat.xcarchive/Info.plist
)"
APP_BUILD_NUMBER="$(
  plutil -extract ApplicationProperties.CFBundleVersion raw \
    -o - build/CaixChat.xcarchive/Info.plist
)"

FASTLANE_SKIP_UPDATE_CHECK=1 \
ASC_KEY_PATH="$ASC_KEY_PATH" \
ASC_KEY_ID="$ASC_KEY_ID" \
ASC_ISSUER_ID="$ASC_ISSUER_ID" \
TEAM_ID="$TEAM_ID" \
BUNDLE_ID="$BUNDLE_ID" \
APP_VERSION="$APP_VERSION" \
APP_BUILD_NUMBER="$APP_BUILD_NUMBER" \
INTERNAL_TESTER_EMAIL="$INTERNAL_TESTER_EMAIL" \
fastlane ios assign_internal_testflight
