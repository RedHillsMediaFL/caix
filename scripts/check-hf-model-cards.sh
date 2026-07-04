#!/usr/bin/env bash
# Fetch live Hugging Face model cards for manifest repos and run the public-copy guard.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-hf-model-cards.sh [options]

Options:
  --manifest <path>  TSV manifest. Default: benchmarks/MANIFEST.tsv.
  --revision <ref>   Hub ref to fetch. Default: main.
  --cards-dir <dir>  Read local card fixtures instead of fetching from the Hub.

Reads README.md files through the Hugging Face CLI unless --cards-dir is set. Does not download model
payloads. Fails when a card has public-copy wording that should not ship, omits the support link,
omits the card-v2 evidence table, lacks a required status block, or claims ready/verified status
without parity and speed evidence.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MANIFEST="$REPO_DIR/benchmarks/MANIFEST.tsv"
REVISION="main"
CARDS_DIR=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="${2:?}"; shift 2 ;;
    --revision) REVISION="${2:?}"; shift 2 ;;
    --cards-dir) CARDS_DIR="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$MANIFEST" ]] || { echo "error: manifest not found: $MANIFEST" >&2; exit 2; }
if [[ -n "$CARDS_DIR" ]]; then
  [[ -d "$CARDS_DIR" ]] || { echo "error: cards dir not found: $CARDS_DIR" >&2; exit 2; }
else
  command -v hf >/dev/null 2>&1 || { echo "error: hf CLI is required" >&2; exit 2; }
fi

export HF_HOME="${HF_HOME:-/Volumes/SSD/hf-cache}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:Cannot enable progress bars}"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-hf-model-cards.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

manifest_repos="$tmpdir/manifest-repos.txt"
cards_dir="$tmpdir/cards"
mkdir -p "$cards_dir"

awk -F '\t' '$1 != "" && $1 != "repo" && $1 !~ /^#/ { print $1 }' "$MANIFEST" \
  | sort -u > "$manifest_repos"

if [[ ! -s "$manifest_repos" ]]; then
  echo "error: manifest has no repos" >&2
  exit 1
fi

missing=0
while IFS= read -r repo; do
  [[ -z "$repo" ]] && continue
  out="$cards_dir/${repo//\//__}.README.md"
  if [[ -n "$CARDS_DIR" ]]; then
    flat="$CARDS_DIR/${repo//\//__}.README.md"
    nested="$CARDS_DIR/$repo/README.md"
    if [[ -f "$flat" ]]; then
      cp "$flat" "$out"
    elif [[ -f "$nested" ]]; then
      cp "$nested" "$out"
    else
      echo "error: missing local model card fixture: $repo ($flat or $nested)" >&2
      missing=1
    fi
  else
    repo_dir="$tmpdir/download/${repo//\//__}"
    if ! hf download "$repo" README.md --revision "$REVISION" --local-dir "$repo_dir" --quiet >/dev/null; then
      echo "error: missing or unreadable model card: $repo@$REVISION" >&2
      missing=1
    elif [[ ! -f "$repo_dir/README.md" ]]; then
      echo "error: README.md missing after download: $repo@$REVISION" >&2
      missing=1
    else
      cp "$repo_dir/README.md" "$out"
    fi
  fi
done < "$manifest_repos"

if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

support_missing=0
while IFS= read -r repo; do
  [[ -z "$repo" ]] && continue
  card="$cards_dir/${repo//\//__}.README.md"
  if ! rg -q 'https://redhillsmediafl[.]com/open-source|redhillsmediafl[.]com/open-source' "$card"; then
    echo "error: support link missing from model card: $repo@$REVISION" >&2
    support_missing=1
  fi
done < "$manifest_repos"

if [[ "$support_missing" -ne 0 ]]; then
  exit 1
fi

manifest_rows="$tmpdir/manifest-rows.tsv"
awk -F '\t' '
  $1 != "" && $1 != "repo" && $1 !~ /^#/ {
    print $1 "\t" $3 "\t" $4 "\t" $5 "\t" $6
  }
' "$MANIFEST" > "$manifest_rows"

contract_fail=0
while IFS=$'\t' read -r repo kind mode status notes; do
  [[ -z "$repo" ]] && continue
  card="$cards_dir/${repo//\//__}.README.md"
  lower="$(tr '[:upper:]' '[:lower:]' < "$card")"

  require_text() {
    local pattern="$1"
    local message="$2"
    if ! printf '%s\n' "$lower" | rg -q "$pattern"; then
      echo "error: $message: $repo@$REVISION" >&2
      contract_fail=1
    fi
  }

  require_text '^library_name:[[:space:]]*caix$' "model card front matter must declare library_name: caix"
  require_text '^## download$' "model card must include a Download section"
  require_text '^## license$' "model card must include a License section"
  require_text '[|][[:space:]]*base model[[:space:]]*[|]' "model card must include Base model evidence"
  require_text '[|][[:space:]]*format[[:space:]]*[|]' "model card must include Format evidence"
  require_text '[|][[:space:]]*quant[[:space:]]*[|]' "model card must include Quant evidence"
  require_text '[|][[:space:]]*context[[:space:]]*[|]' "model card must include Context evidence"
  require_text '[|][[:space:]]*runtime[[:space:]]*[|]' "model card must include Runtime evidence"
  require_text '[|][[:space:]]*license[[:space:]]*[|]' "model card must include License evidence"

  ready_claim=0
  if printf '%s\n' "$lower" | rg -q 'ready-to-run|verified in caix'; then
    ready_claim=1
  fi
  if [[ "$ready_claim" -eq 1 ]]; then
    require_text 'parity evidence|parity[[:space:]]+gate|teacher-forced|hf-fp16|1:1[[:space:]]+parity' \
      "ready/verified card must cite parity evidence"
    require_text 'speed evidence|benchmark evidence|raw benchmark|decode speed|performance evidence' \
      "ready/verified card must cite speed evidence"
  fi

  case "$status" in
    component_only|blocked_runtime)
      if [[ "$ready_claim" -eq 1 ]]; then
        echo "error: component/blocked card must not claim ready-to-run or verified status: $repo@$REVISION" >&2
        contract_fail=1
      fi
      ;;
  esac

  if [[ "$kind" == "staged" || "$status" == "component_only" ]]; then
    require_text 'caix-status-label' "component/staged card must include a caix-status-label block"
    require_text 'component-only|component only|needs-test|needs test|not a standalone|do not test alone|matching target|requires distributed|distributed hardware|hardware smoke' \
      "component/staged card must be clearly labeled"
  fi
  if [[ "$kind" == "mtp" ]]; then
    require_text 'caix-status-label' "MTP/speculative card must include a caix-status-label block"
    require_text 'mtp|speculative|eagle|target[+ -]?draft|target and draft|target-plus-draft|matching target|standalone target|compare against standalone' \
      "MTP/speculative card must identify the target+draft package and matching target context"
  fi
  if [[ "$status" == "blocked_runtime" ]]; then
    require_text 'caix-status-label' "blocked card must include a caix-status-label block"
    require_text 'blocked|do not test|rebuild|runtime' "blocked card must be clearly labeled"
  fi
  if [[ "$notes" == *"local stdout instability"* || "$notes" == *"publish only if"* || "$notes" == *"nondeterministic"* || "$notes" == *"non-1:1"* ]]; then
    if [[ "$ready_claim" -eq 1 ]]; then
      echo "error: instability-gated card must not claim ready-to-run or verified status: $repo@$REVISION" >&2
      contract_fail=1
    fi
    require_text 'caix-status-label' "instability-gated card must include a caix-status-label block"
    require_text 'stability|determinism|nondeterministic|non-1:1|needs-test|needs test|blocked|do not publish' \
      "instability-gated card must be clearly labeled"
  fi
done < "$manifest_rows"

if [[ "$contract_fail" -ne 0 ]]; then
  exit 1
fi

set +e
guard_output="$("$SCRIPT_DIR/check-public-copy.sh" "$cards_dir" 2>&1)"
guard_status=$?
set -e
if [[ "$guard_status" -ne 0 ]]; then
  printf '%s\n' "$guard_output" \
    | sed "s#$cards_dir/##g" \
    | sed 's#^\([^:]*\)__\([^:]*\)\.README\.md:#\1/\2 README.md:#'
  exit "$guard_status"
fi

count="$(wc -l < "$manifest_repos" | tr -d ' ')"
echo "hf model cards ok: $count cards checked at $REVISION"
