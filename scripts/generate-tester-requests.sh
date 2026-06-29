#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/generate-tester-requests.sh [options]

Options:
  --manifest <path>   TSV manifest. Default: benchmarks/MANIFEST.tsv.
  --revisions <path>  Optional repo<TAB>revision TSV. Default: benchmarks/revisions.tsv when present.
  --raw-dir <path>    Raw benchmark root. Default: benchmarks/raw.
  --out <path>        Markdown output path. Default: stdout.

Writes a blunt tester request sheet. Does not download models.
When --raw-dir is inside this repository, only tracked raw metadata is counted as existing evidence.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MANIFEST="$REPO_DIR/benchmarks/MANIFEST.tsv"
REVISIONS=""
RAW_DIR="$REPO_DIR/benchmarks/raw"
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="${2:?}"; shift 2 ;;
    --revisions) REVISIONS="${2:?}"; shift 2 ;;
    --raw-dir) RAW_DIR="${2:?}"; shift 2 ;;
    --out) OUT="${2:?}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -f "$MANIFEST" ]] || { echo "error: manifest not found: $MANIFEST" >&2; exit 2; }
if [[ -z "$REVISIONS" && -f "$REPO_DIR/benchmarks/revisions.tsv" ]]; then
  REVISIONS="$REPO_DIR/benchmarks/revisions.tsv"
fi
if [[ -n "$REVISIONS" ]]; then
  [[ -f "$REVISIONS" ]] || { echo "error: revisions file not found: $REVISIONS" >&2; exit 2; }
fi

TMP_FILES=()
cleanup() {
  if [[ ${#TMP_FILES[@]} -gt 0 ]]; then
    rm -f "${TMP_FILES[@]}"
  fi
}
trap cleanup EXIT

manifest_label="${MANIFEST#$REPO_DIR/}"
if [[ -n "$REVISIONS" ]]; then
  revisions_label="${REVISIONS#$REPO_DIR/}"
else
  revisions_label="none"
fi
raw_label="${RAW_DIR#$REPO_DIR/}"

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

tracked_or_local_metadata() {
  [[ -d "$RAW_DIR" ]] || return 0

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

EVIDENCE_TSV="$(mktemp "${TMPDIR:-/tmp}/caix-tester-evidence.XXXXXX")"
TMP_FILES+=("$EVIDENCE_TSV")
: > "$EVIDENCE_TSV"

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
    "$repo" "$revision" "$name" "$mode" "$measured" "$raw_dir_label" >> "$EVIDENCE_TSV"
done < <(tracked_or_local_metadata | sort)

write_markdown() {
  cat <<EOF
# Tester Requests

Generated from \`$manifest_label\`.
Revision source: \`$revisions_label\`.
Raw evidence source: \`$raw_label\`.

No speed claims without raw logs. Use the exact revision in the table. Keep prompts, token budget,
temperature, streaming mode, warmup count, measured run count, and chat-template mode unchanged.

## Ready Benchmark Requests

| repo | revision | local dir | request | notes |
|---|---|---|---|---|
EOF

  awk -F '\t' -v revfile="${REVISIONS:-/dev/null}" -v evidence="${EVIDENCE_TSV:-/dev/null}" '
    BEGIN {
      while ((getline line < revfile) > 0) {
        split(line, parts, "\t")
        if (parts[1] != "" && parts[2] != "") {
          revision[parts[1]] = parts[2]
        }
      }
      close(revfile)
      while ((getline line < evidence) > 0) {
        split(line, parts, "\t")
        if (parts[1] != "" && parts[2] != "") {
          evidence_revision[parts[1]] = parts[2]
        }
      }
      close(evidence)
    }
    function cell(value) {
      gsub(/\|/, "/", value)
      gsub(/\r/, "", value)
      gsub(/\t/, " ", value)
      return value
    }
    function sha(repo) {
      return (repo in revision) ? revision[repo] : "<record-before-testing>"
    }
    function ready_mode(mode) {
      return mode == "decode" || mode == "speculative" || mode == "eagle" || mode == "eagle-mtp"
    }
    function request_for(mode) {
      if (mode == "speculative") return "classic speculative load, generation, benchmark"
      if (mode == "eagle" || mode == "eagle-mtp") return "EAGLE MTP load, generation, benchmark"
      return "load, generation, benchmark"
    }
    $1 == "" || $1 == "repo" || $1 ~ /^#/ { next }
    ready_mode($4) && $5 == "eligible" && !($1 in evidence_revision) {
      request = request_for($4)
      printf "| `%s` | `%s` | `%s` | %s | %s |\n",
        cell($1), sha($1), cell($2), request, cell($6)
    }
  ' "$MANIFEST"

  cat <<'EOF'

## Existing Raw Evidence

| repo | revision | local dir | mode | measured runs | raw dir |
|---|---|---|---|---|---|
EOF

  awk -F '\t' -v evidence="${EVIDENCE_TSV:-/dev/null}" '
    BEGIN {
      while ((getline line < evidence) > 0) {
        split(line, parts, "\t")
        if (parts[1] != "" && parts[2] != "") {
          evidence_revision[parts[1]] = parts[2]
          evidence_name[parts[1]] = parts[3]
          evidence_mode[parts[1]] = parts[4]
          evidence_runs[parts[1]] = parts[5]
          evidence_dir[parts[1]] = parts[6]
        }
      }
      close(evidence)
    }
    function cell(value) {
      gsub(/\|/, "/", value)
      gsub(/\r/, "", value)
      gsub(/\t/, " ", value)
      return value
    }
    function ready_mode(mode) {
      return mode == "decode" || mode == "speculative" || mode == "eagle" || mode == "eagle-mtp"
    }
    $1 == "" || $1 == "repo" || $1 ~ /^#/ { next }
    ready_mode($4) && $5 == "eligible" && ($1 in evidence_revision) {
      name = evidence_name[$1] != "" ? evidence_name[$1] : $2
      printf "| `%s` | `%s` | `%s` | `%s` | %s | `%s` |\n",
        cell($1), cell(evidence_revision[$1]), cell(name), cell(evidence_mode[$1]),
        cell(evidence_runs[$1]), cell(evidence_dir[$1])
    }
  ' "$MANIFEST"

  cat <<'EOF'

## Manual Or Component Requests

| repo | revision | local dir | request | notes |
|---|---|---|---|---|
EOF

  awk -F '\t' -v revfile="${REVISIONS:-/dev/null}" '
    BEGIN {
      while ((getline line < revfile) > 0) {
        split(line, parts, "\t")
        if (parts[1] != "" && parts[2] != "") {
          revision[parts[1]] = parts[2]
        }
      }
      close(revfile)
    }
    function cell(value) {
      gsub(/\|/, "/", value)
      gsub(/\r/, "", value)
      gsub(/\t/, " ", value)
      return value
    }
    function sha(repo) {
      return (repo in revision) ? revision[repo] : "<record-before-testing>"
    }
    function ready_mode(mode) {
      return mode == "decode" || mode == "speculative" || mode == "eagle" || mode == "eagle-mtp"
    }
    $1 == "" || $1 == "repo" || $1 ~ /^#/ { next }
    !(ready_mode($4) && $5 == "eligible") {
      request = ($3 == "draft") ? "component; do not test alone" : "manual target plus draft"
      printf "| `%s` | `%s` | `%s` | %s | %s |\n",
        cell($1), sha($1), cell($2), request, cell($6)
    }
  ' "$MANIFEST"

  cat <<'EOF'

## Run Template

Set one row's values:

```bash
REPO=<repo-from-table>
REVISION=<revision-from-table>
NAME=<local-dir-from-table>
```

Install one payload:

```bash
scripts/check-disk-pressure.sh --path /Volumes/SSD --floor-gib 500
mkdir -p models/exports
hf download "$REPO" \
  --revision "$REVISION" \
  --local-dir "models/exports/$NAME"
```

Verify:

```bash
CAIX_BIN=${CAIX_BIN:-.build/release/caix}
MODEL="models/exports/$NAME"

"$CAIX_BIN" inspect --model "$MODEL"
"$CAIX_BIN" run \
  --model "$MODEL" \
  --prompt "Name one primary color." \
  --max-tokens 32 \
  --temperature 0 \
  --verbose
```

Benchmark:

```bash
scripts/benchmark-model.sh \
  --model "models/exports/$NAME" \
  --name "$NAME" \
  --repo "$REPO" \
  --repo-revision "$REVISION" \
  --prompt "Write one factual sentence about local inference on Apple silicon." \
  --max-tokens 128 \
  --temperature 0 \
  --warmup 1 \
  --runs 3
```

For classic speculative rows, add the draft bundle:

```bash
scripts/benchmark-model.sh \
  --model "models/exports/$NAME" \
  --draft "models/exports/$NAME/draft" \
  --name "$NAME" \
  --repo "$REPO" \
  --repo-revision "$REVISION" \
  --prompt "Write one factual sentence about local inference on Apple silicon." \
  --max-tokens 128 \
  --temperature 0 \
  --warmup 1 \
  --runs 3
```

For EAGLE MTP rows, benchmark the package:

```bash
scripts/benchmark-eagle.sh \
  --package "models/exports/$NAME" \
  --name "$NAME" \
  --repo "$REPO" \
  --repo-revision "$REVISION" \
  --prompt "Write one factual sentence about local inference on Apple silicon." \
  --max-tokens 128 \
  --warmup 1 \
  --runs 3
```

Report the fields in `docs/TESTING.md`. Send the raw benchmark directory. Remove only the payload
you installed:

```bash
rm -rf "models/exports/$NAME"
scripts/check-disk-pressure.sh --path /Volumes/SSD --floor-gib 500
```
EOF
}

if [[ -n "$OUT" ]]; then
  mkdir -p "$(dirname "$OUT")"
  tmp="$OUT.tmp.$$"
  TMP_FILES+=("$tmp")
  write_markdown > "$tmp"
  mv "$tmp" "$OUT"
  echo "$OUT"
else
  write_markdown
fi
