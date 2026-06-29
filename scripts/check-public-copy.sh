#!/usr/bin/env bash
# Fail public docs/pages that drift into benchmark placeholders, hype, or vague model-size language.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-public-copy.sh [path ...]

Default paths:
  README.md CHANGELOG.md docs web Formula

Checks public-facing text only. Do not point this at internal coordination logs.
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ "$#" -gt 0 ]; then
  paths=("$@")
else
  paths=(README.md CHANGELOG.md docs web Formula)
fi

existing=()
for path in "${paths[@]}"; do
  if [ -e "$path" ]; then
    existing+=("$path")
  fi
done

if [ "${#existing[@]}" -eq 0 ]; then
  echo "error: no input paths exist" >&2
  exit 2
fi

fail=0

scan() {
  local label="$1"
  local pattern="$2"
  if rg -n -i --glob '!benchmarks/raw/**' --glob '!benchmarks/reports/**' "$pattern" "${existing[@]}"; then
    echo "error: $label" >&2
    fail=1
  fi
}

scan "benchmark placeholder or unsupported public speed claim" \
  'benchmark pending|coming soon|fastest|blazing|guaranteed|100%[[:space:]]+(compatible|support(ed)?|coverage|accurate|accuracy|working|faster|speed|safe|verified)'
scan "raw benchmark speed number in public copy" \
  '[0-9]+([.][0-9]+)?[[:space:]]*tok/s'
scan "marketing/hype wording" \
  'revolutionary|game[- ]changer|world[- ]class|best[- ]in[- ]class|magic|gimmick'
scan "support gimmick wording" \
  'donat(e|ion)?|sticker'
scan "vague model-size wording" \
  'large model|large chat|dense large|\blarger\b|\blargest\b'
scan "bare large wording; use parameter count, disk size, memory, and license instead" \
  '\blarge\b'
scan "unsafe export cleanup command; use scripts/remove-export.sh" \
  'rm[[:space:]]+-rf[^[:cntrl:]]*models/exports'

if [ "$fail" -ne 0 ]; then
  exit 1
fi

echo "public copy ok"
