#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-release-version.sh [--coreai-stable] [--dev-ok] <version>

Validates caix release versions.
While Core AI is beta, release versions must stay below 1.0.0.
USAGE
}

COREAI_STABLE="${caix_coreai_stable:-0}"
DEV_OK=0
VERSION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --coreai-stable) COREAI_STABLE=1; shift ;;
    --dev-ok) DEV_OK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [[ -n "$VERSION" ]]; then
        echo "error: only one version is accepted" >&2
        exit 2
      fi
      VERSION="$1"
      shift
      ;;
  esac
done

[[ -n "$VERSION" ]] || { echo "error: version is required" >&2; usage >&2; exit 2; }

RAW_VERSION="${VERSION#v}"
if [[ ! "$RAW_VERSION" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(-[0-9A-Za-z][0-9A-Za-z.-]*)?$ ]]; then
  echo "error: version must look like v0.2.0-beta or 0.2.0" >&2
  exit 2
fi

MAJOR="${BASH_REMATCH[1]}"
if (( MAJOR >= 1 )) && [[ "$COREAI_STABLE" != "1" ]]; then
  echo "error: Core AI is beta; keep caix releases below 1.0.0" >&2
  exit 1
fi

if [[ "$DEV_OK" != "1" && "$RAW_VERSION" == *dev* ]]; then
  echo "error: release version must not contain dev" >&2
  exit 1
fi

echo "release version ok: v$RAW_VERSION"
