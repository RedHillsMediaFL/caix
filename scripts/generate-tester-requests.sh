#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/generate-tester-requests.sh [options]

Options:
  --manifest <path>   TSV manifest. Default: benchmarks/MANIFEST.tsv.
  --revisions <path>  Optional repo<TAB>revision TSV. Default: benchmarks/revisions.tsv when present.
  --out <path>        Markdown output path. Default: stdout.

Writes a blunt tester request sheet. Does not download models.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MANIFEST="$REPO_DIR/benchmarks/MANIFEST.tsv"
REVISIONS=""
OUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="${2:?}"; shift 2 ;;
    --revisions) REVISIONS="${2:?}"; shift 2 ;;
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

manifest_label="${MANIFEST#$REPO_DIR/}"
if [[ -n "$REVISIONS" ]]; then
  revisions_label="${REVISIONS#$REPO_DIR/}"
else
  revisions_label="none"
fi

write_markdown() {
  cat <<EOF
# Tester Requests

Generated from \`$manifest_label\`.
Revision source: \`$revisions_label\`.

No speed claims without raw logs. Use the exact revision in the table. Keep prompts, token budget,
temperature, streaming mode, warmup count, measured run count, and chat-template mode unchanged.

## Ready Benchmark Requests

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
    function request_for(mode) {
      if (mode == "speculative") return "classic speculative load, generation, benchmark"
      if (mode == "eagle" || mode == "eagle-mtp") return "EAGLE MTP load, generation, benchmark"
      return "load, generation, benchmark"
    }
    $1 == "" || $1 == "repo" || $1 ~ /^#/ { next }
    ready_mode($4) && $5 == "eligible" {
      request = request_for($4)
      printf "| `%s` | `%s` | `%s` | %s | %s |\n",
        cell($1), sha($1), cell($2), request, cell($6)
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
REPO=redhillsmediafl/rhm-qwen3-4b-caix
REVISION=85159782da417ce077fad5948a09f654b8d81675
NAME=qwen3-4b-coreai
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
  trap 'rm -f "$tmp"' EXIT
  write_markdown > "$tmp"
  mv "$tmp" "$OUT"
  trap - EXIT
  echo "$OUT"
else
  write_markdown
fi
