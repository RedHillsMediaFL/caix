#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-brew-distributed.sh [--caix <path>]

Checks the Homebrew-installed caix surface needed before Thunderbolt distributed tests.
It does not start workers or run inference.
USAGE
}

caix_binary="${caix_bin:-caix}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --caix) caix_binary="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) echo "error: unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

command -v "$caix_binary" >/dev/null 2>&1 || {
  echo "error: caix binary not found: $caix_binary" >&2
  exit 1
}

"$caix_binary" --version
"$caix_binary" doctor --no-fail
"$caix_binary" cluster plan --help >/dev/null

echo "brew distributed surface ok"
