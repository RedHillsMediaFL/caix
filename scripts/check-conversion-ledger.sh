#!/usr/bin/env bash
# Check that every registry conversion lane has an explicit publish/block status.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-conversion-ledger.sh [options]

Options:
  --registry <path>  Model registry JSON. Default: models/registry.json.
  --ledger <path>    Conversion ledger TSV. Default: docs/CONVERSION_LEDGER.tsv.
  --manifest <path>  Benchmark manifest TSV. Default: benchmarks/MANIFEST.tsv.

Does not download models. Fails when registry conversion_order entries are missing from the ledger,
when source repos drift, or when published repos are missing from the benchmark manifest.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

REGISTRY="$REPO_DIR/models/registry.json"
LEDGER="$REPO_DIR/docs/CONVERSION_LEDGER.tsv"
MANIFEST="$REPO_DIR/benchmarks/MANIFEST.tsv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --registry) REGISTRY="${2:?}"; shift 2 ;;
    --ledger) LEDGER="${2:?}"; shift 2 ;;
    --manifest) MANIFEST="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$REGISTRY" ]] || { echo "error: registry not found: $REGISTRY" >&2; exit 2; }
[[ -f "$LEDGER" ]] || { echo "error: ledger not found: $LEDGER" >&2; exit 2; }
[[ -f "$MANIFEST" ]] || { echo "error: manifest not found: $MANIFEST" >&2; exit 2; }
command -v jq >/dev/null 2>&1 || { echo "error: jq is required" >&2; exit 2; }

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-conversion-ledger.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

registry_tsv="$tmpdir/registry.tsv"
registry_keys="$tmpdir/registry-keys.txt"
ledger_keys="$tmpdir/ledger-keys.txt"
manifest_repos="$tmpdir/manifest-repos.txt"

jq -e '
  .models as $models
  | (.conversion_order | type == "array")
  and all(.conversion_order[]; ($models[.] and (($models[.].hf_repo // "") != "")))
' "$REGISTRY" >/dev/null

jq -r '.conversion_order[] as $key | [$key, .models[$key].hf_repo] | @tsv' "$REGISTRY" \
  > "$registry_tsv"

cut -f1 "$registry_tsv" | sort -u > "$registry_keys"

awk -F '\t' '
  NR == 1 {
    expected = "model_key\tsource_repo\tstatus\tpublished_repo\tnext_step"
    if ($0 != expected) {
      printf "error: bad ledger header: %s\n", $0 > "/dev/stderr"
      exit 2
    }
    next
  }
  NF != 5 {
    printf "error: bad ledger row %d: expected 5 tab-separated fields\n", NR > "/dev/stderr"
    exit 2
  }
  $1 == "" || $2 == "" || $3 == "" || $4 == "" || $5 == "" {
    printf "error: empty ledger field on row %d\n", NR > "/dev/stderr"
    exit 2
  }
  { print $1 }
' "$LEDGER" | sort -u > "$ledger_keys"

missing="$(comm -23 "$registry_keys" "$ledger_keys")"
if [[ -n "$missing" ]]; then
  echo "error: registry conversion entries missing from ledger:" >&2
  printf '%s\n' "$missing" >&2
  exit 1
fi

extra="$(comm -13 "$registry_keys" "$ledger_keys")"
if [[ -n "$extra" ]]; then
  echo "error: ledger entries not present in registry conversion_order:" >&2
  printf '%s\n' "$extra" >&2
  exit 1
fi

awk -F '\t' '$1 != "" && $1 != "repo" && $1 !~ /^#/ { print $1 }' "$MANIFEST" \
  | sort -u > "$manifest_repos"

awk -F '\t' -v registry="$registry_tsv" -v manifest="$manifest_repos" '
  BEGIN {
    while ((getline line < registry) > 0) {
      split(line, parts, "\t")
      source[parts[1]] = parts[2]
    }
    close(registry)
    while ((getline repo < manifest) > 0) {
      manifest_repo[repo] = 1
    }
    close(manifest)
    allowed["published"] = 1
    allowed["component_published"] = 1
    allowed["blocked_runtime"] = 1
    allowed["blocked_architecture"] = 1
    allowed["blocked_access"] = 1
    allowed["blocked_host_fit"] = 1
    allowed["needs_source_download"] = 1
  }
  NR == 1 { next }
  {
    key = $1
    repo = $2
    status = $3
    published = $4
    next_step = $5
    if (!(status in allowed)) {
      printf "error: unsupported ledger status for %s: %s\n", key, status > "/dev/stderr"
      fail = 1
    }
    if (repo != source[key]) {
      printf "error: source repo mismatch for %s: ledger=%s registry=%s\n", key, repo, source[key] > "/dev/stderr"
      fail = 1
    }
    if (status ~ /^blocked_/ || status == "needs_source_download") {
      if (published != "-") {
        printf "error: blocked ledger row has published repo for %s: %s\n", key, published > "/dev/stderr"
        fail = 1
      }
    } else {
      if (published == "-") {
        printf "error: published ledger row missing published repo for %s\n", key > "/dev/stderr"
        fail = 1
      } else {
        n = split(published, repos, ",")
        for (i = 1; i <= n; i++) {
          if (!(repos[i] in manifest_repo)) {
            printf "error: published repo missing from benchmark manifest for %s: %s\n", key, repos[i] > "/dev/stderr"
            fail = 1
          }
        }
      }
    }
    if (next_step == "-") {
      printf "error: next_step must be explicit for %s\n", key > "/dev/stderr"
      fail = 1
    }
  }
  END { exit fail ? 1 : 0 }
' "$LEDGER"

count="$(wc -l < "$registry_keys" | tr -d ' ')"
echo "conversion ledger ok: $count registry conversion lanes accounted for"
