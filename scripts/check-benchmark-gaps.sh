#!/usr/bin/env bash
# Report manifest-eligible benchmark rows that still lack committed raw evidence.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-benchmark-gaps.sh [options]

Options:
  --manifest <path>  TSV manifest. Default: benchmarks/MANIFEST.tsv.
  --raw-dir <path>   Raw benchmark root. Default: benchmarks/raw.
  --strict           Exit non-zero when an eligible manifest row lacks raw evidence.

Does not run models, download payloads, or contact Hugging Face.
Counts only measured raw logs whose metadata and summary are committed or locally available.
Matches evidence by repo and benchmark_mode so decode, classic speculative, and EAGLE rows stay
separate.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MANIFEST="$REPO_DIR/benchmarks/MANIFEST.tsv"
RAW_DIR="$REPO_DIR/benchmarks/raw"
STRICT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="${2:?}"; shift 2 ;;
    --raw-dir) RAW_DIR="${2:?}"; shift 2 ;;
    --strict) STRICT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$MANIFEST" ]] || { echo "error: manifest not found: $MANIFEST" >&2; exit 2; }
[[ -d "$RAW_DIR" ]] || { echo "error: raw benchmark directory not found: $RAW_DIR" >&2; exit 2; }

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-benchmark-gaps.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT
evidence="$tmpdir/evidence.tsv"
: > "$evidence"

metadata_value() {
  local key="$1"
  local file="$2"
  awk -v key="$key" '
    index($0, key "=") == 1 {
      sub("^[^=]*=", "")
      print
      exit
    }
  ' "$file"
}

canonical_benchmark_mode() {
  case "$1" in
    eagle) printf 'eagle-mtp' ;;
    *) printf '%s' "$1" ;;
  esac
}

repo_relative_path() {
  local path="$1"
  local abs
  abs="$(cd "$path" && pwd)"
  case "$abs/" in
    "$REPO_DIR"/*)
      printf '%s\n' "${abs#$REPO_DIR/}"
      ;;
    *)
      return 1
      ;;
  esac
}

raw_dir_has_git_changes() {
  local raw_dir="$1"
  local rel
  rel="$(repo_relative_path "$raw_dir")" || return 1
  [[ -n "$(git -C "$REPO_DIR" status --porcelain -- "$rel")" ]]
}

tracked_or_local_metadata() {
  local raw_abs raw_rel
  raw_abs="$(cd "$RAW_DIR" && pwd)"
  case "$raw_abs/" in
    "$REPO_DIR"/*)
      raw_rel="${raw_abs#$REPO_DIR/}"
      git -C "$REPO_DIR" ls-files -- "$raw_rel" \
        | awk -v repo="$REPO_DIR" '/\/metadata[.]txt$/ { print repo "/" $0 }'
      ;;
    *)
      find "$RAW_DIR" -type f -name metadata.txt -print
      ;;
  esac
}

while IFS= read -r metadata; do
  dir="${metadata%/metadata.txt}"
  [[ "$(basename "$dir")" == *-suite ]] && continue

  summary="$dir/summary.tsv"
  [[ -f "$summary" ]] || continue
  raw_dir_has_git_changes "$dir" && continue

  repo="$(metadata_value repo "$metadata")"
  revision="$(metadata_value repo_revision "$metadata")"
  name="$(metadata_value name "$metadata")"
  mode="$(canonical_benchmark_mode "$(metadata_value benchmark_mode "$metadata")")"
  runs="$(metadata_value runs "$metadata")"
  [[ "$repo" == redhillsmediafl/rhm-*-caix ]] || continue
  [[ "$revision" =~ ^[0-9a-f]{40}$ ]] || continue
  [[ "$runs" =~ ^[1-9][0-9]*$ ]] || continue

  measured="$(awk -F '\t' 'NR > 1 && $1 == "measured" && $3 == "ok" { n++ } END { print n + 0 }' "$summary")"
  failed="$(awk -F '\t' 'NR > 1 && $1 == "measured" && $3 != "ok" { n++ } END { print n + 0 }' "$summary")"
  [[ "$measured" == "$runs" && "$failed" == "0" ]] || continue

  raw_dir_label="${dir#$REPO_DIR/}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$repo" "$mode" "$revision" "$name" "$measured" "$raw_dir_label" >> "$evidence"
done < <(tracked_or_local_metadata | sort)

awk -F '\t' -v evidence="$evidence" -v strict="$STRICT" '
  function canonical_mode(mode) {
    return mode == "eagle" ? "eagle-mtp" : mode
  }
  function ready_mode(mode) {
    return mode == "decode" || mode == "speculative" || mode == "eagle-mtp"
  }
  function cell(value) {
    gsub(/\r/, "", value)
    gsub(/\t/, " ", value)
    return value
  }
  BEGIN {
    while ((getline line < evidence) > 0) {
      split(line, parts, "\t")
      key = parts[1] SUBSEP canonical_mode(parts[2])
      if (!(key in evidence_revision)) {
        evidence_revision[key] = parts[3]
        evidence_runs[key] = parts[5]
        evidence_dir[key] = parts[6]
      }
    }
    close(evidence)
  }
  $1 == "" || $1 == "repo" || $1 ~ /^#/ { next }
  {
    mode = canonical_mode($4)
    status = $5
    if (ready_mode(mode) && status == "eligible") {
      eligible++
      key = $1 SUBSEP mode
      if (key in evidence_revision) {
        measured++
      } else {
        pending++
        pending_rows[pending] = sprintf("%s\t%s\t%s\t%s", cell($1), cell($2), cell(mode), cell($6))
      }
    } else {
      noneligible++
    }
  }
  END {
    if (pending == 0) {
      printf "benchmark gap audit ok: %d/%d eligible rows have committed measured raw evidence; noneligible=%d\n",
        measured, eligible, noneligible
      exit 0
    }

    printf "benchmark gap audit: %d/%d eligible rows have committed measured raw evidence; pending=%d; noneligible=%d\n",
      measured, eligible, pending, noneligible
    print "pending_repo\tlocal_dir\tbenchmark_mode\tnotes"
    for (i = 1; i <= pending; i++) {
      print pending_rows[i]
    }

    if (strict == "1") {
      exit 1
    }
  }
' "$MANIFEST"
