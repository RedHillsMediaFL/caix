#!/usr/bin/env bash
# Fail before local model, benchmark, or conversion work starts on a low-free-space volume.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-disk-pressure.sh [options]

Options:
  --path <path>       Filesystem path to check. Default: /Volumes/SSD.
  --floor-gib <n>    Required free GiB. Default: 500.
  --quiet            Print only errors.
  --json             Emit machine-readable status.

Exit 0 when free space is at or above the floor. Exit 1 when below the floor.
USAGE
}

PATH_TO_CHECK="/Volumes/SSD"
FLOOR_GIB=500
QUIET=0
JSON=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --path)
      PATH_TO_CHECK="${2:?missing value for --path}"
      shift 2
      ;;
    --floor-gib)
      FLOOR_GIB="${2:?missing value for --floor-gib}"
      shift 2
      ;;
    --quiet)
      QUIET=1
      shift
      ;;
    --json)
      JSON=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 2
      ;;
  esac
done

case "$FLOOR_GIB" in
  ''|*[!0-9]*)
    echo "error: --floor-gib must be a non-negative integer" >&2
    exit 2
    ;;
esac

if [ ! -e "$PATH_TO_CHECK" ]; then
  echo "error: path not found: $PATH_TO_CHECK" >&2
  exit 2
fi

free_gib="$(df -g "$PATH_TO_CHECK" 2>/dev/null | awk 'NR==2 { print $4 }')"
if [ -z "$free_gib" ]; then
  echo "error: could not read filesystem free space for $PATH_TO_CHECK" >&2
  exit 2
fi

status="ok"
rc=0
if [ "$free_gib" -lt "$FLOOR_GIB" ]; then
  status="low"
  rc=1
fi

if [ "$JSON" -eq 1 ]; then
  printf '{"path":"%s","free_gib":%s,"floor_gib":%s,"status":"%s"}\n' \
    "$PATH_TO_CHECK" "$free_gib" "$FLOOR_GIB" "$status"
elif [ "$QUIET" -eq 0 ]; then
  if [ "$rc" -eq 0 ]; then
    printf 'disk ok: %s GiB free at %s (floor %s GiB)\n' \
      "$free_gib" "$PATH_TO_CHECK" "$FLOOR_GIB"
  else
    printf 'error: only %s GiB free at %s; floor is %s GiB\n' \
      "$free_gib" "$PATH_TO_CHECK" "$FLOOR_GIB" >&2
  fi
fi

exit "$rc"
