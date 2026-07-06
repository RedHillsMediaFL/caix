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
payloads. Fails when a card has public-copy wording that should not ship or omits the production
model-card contract: caix metadata, a small RHM logo, install/run instructions, user-facing specs,
license, and support footer. Internal validation numbers and build/test process details belong in
MANIFEST/ledger files, not public cards.
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

  duplicate_tags="$(
    awk '
      BEGIN { front = 0; tags = 0 }
      /^---[[:space:]]*$/ {
        if (front == 0) { front = 1; next }
        exit
      }
      front && /^tags:[[:space:]]*$/ { tags = 1; next }
      front && tags && /^[[:space:]]*-/ {
        tag = $0
        sub(/^[[:space:]]*-[[:space:]]*/, "", tag)
        sub(/[[:space:]]*#.*/, "", tag)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", tag)
        if (tag != "") print tolower(tag)
        next
      }
      front && tags && /^[[:alnum:]_-]+:/ { tags = 0 }
    ' "$card" | sort | uniq -d
  )"
  if [[ -n "$duplicate_tags" ]]; then
    echo "error: duplicate front-matter tag(s) in model card: $repo@$REVISION ($(tr '\n' ',' <<<"$duplicate_tags" | sed 's/,$//'))" >&2
    contract_fail=1
  fi

  if printf '%s\n' "$lower" | rg -q 'this[[:space:]]+none[[:space:]]*;'; then
    echo "error: unfilled compression fragment in model card: $repo@$REVISION (This none;)" >&2
    contract_fail=1
  fi

  require_text() {
    local pattern="$1"
    local message="$2"
    if ! printf '%s\n' "$lower" | rg -q "$pattern"; then
      echo "error: $message: $repo@$REVISION" >&2
      contract_fail=1
    fi
  }

  reject_text() {
    local pattern="$1"
    local message="$2"
    if printf '%s\n' "$lower" | rg -q "$pattern"; then
      echo "error: $message: $repo@$REVISION" >&2
      contract_fail=1
    fi
  }

  require_text '^library_name:[[:space:]]*caix$' "model card front matter must declare library_name: caix"
  require_text '^base_model:[[:space:]]*[^[:space:]].*$' "model card front matter must declare base_model"
  require_text '<img[^>]+(redhillsmediafl|red hills media|rhm)[^>]+width="?([8-9][0-9]|1[0-5][0-9])"?|<img[^>]+width="?([8-9][0-9]|1[0-5][0-9])"?[^>]+(redhillsmediafl|red hills media|rhm)' \
    "model card must include a small RHM logo (width 80-159)"
  require_text '^## install[[:space:]]*&[[:space:]]*run$' "model card must include an Install & run section"
  require_text 'brew install redhillsmediafl/caix/caix|brew upgrade redhillsmediafl/caix/caix' \
    "model card must include Homebrew caix install instructions"
  require_text 'caix catalog install[[:space:]]+redhillsmediafl/' \
    "model card must include caix catalog install instructions"
  require_text 'caix serve --model[[:space:]]+' "model card must include caix serve --model instructions"
  require_text '^## at a glance$' "model card must include an At a glance section"
  require_text 'context:.*matches base model' "model card must state context matches base model"
  require_text 'runs on:.*apple silicon' "model card must state Apple silicon support"
  require_text 'base model:' "model card must name the base model in At a glance"
  require_text 'quantization:|precision:|weights:' "model card must state quantization/precision"
  require_text 'download:|disk:|size:' "model card must state download/disk size"
  require_text '^## status$' "model card must include a Status section"
  require_text '^(verified|needs-test|needs test|component|blocked|beta)' \
    "model card status must use verified/needs-test/component/blocked/beta wording"
  require_text '^## license$' "model card must include a License section"
  reject_text 'tested on|validated on|verified on .*mac|mac studio|macbook|mac mini|xcode|os build|sw_vers|build host|build machine|built on|converted on|export host|where we build|when we build|why we build|how we build|coreai-models commit|caix commit|oracle token|determinism count|peak rss|wired memory' \
    "model card must not include build/test device details"

  ready_claim=0
  if printf '%s\n' "$lower" | rg -q 'ready-to-run|ready to run|verified in caix'; then
    ready_claim=1
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
    require_text 'staged|distributed|multi-device|multimodal|understands images|image' \
      "component/staged card must be clearly labeled"
  fi
  if [[ "$kind" == "mtp" ]]; then
    require_text 'mtp|speculative|eagle|target[+ -]?draft|target and draft|target-plus-draft|matching target|standalone target|compare against standalone' \
      "MTP/speculative card must identify the target+draft package and matching target context"
  fi
  if [[ "$status" == "blocked_runtime" ]]; then
    require_text 'blocked|do not test|rebuild|runtime' "blocked card must be clearly labeled"
  fi
  if [[ "$notes" == *"local stdout instability"* || "$notes" == *"publish only if"* || "$notes" == *"nondeterministic"* || "$notes" == *"non-1:1"* ]]; then
    if [[ "$ready_claim" -eq 1 ]]; then
      echo "error: instability-gated card must not claim ready-to-run or verified status: $repo@$REVISION" >&2
        contract_fail=1
    fi
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
