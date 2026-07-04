#!/usr/bin/env bash
# Self-test tester-request evidence handling with local fixtures only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-tester-requests-contract.XXXXXX")"
inside_repo_tmp="$REPO_DIR/.tmp/tester-requests-contract-$$"
cleanup() {
  rm -rf "$tmpdir" "$inside_repo_tmp"
}
trap cleanup EXIT

repo="redhillsmediafl/rhm-fixture-caix"
mtp_repo="redhillsmediafl/rhm-fixture-mtp-caix"
staged_repo="redhillsmediafl/rhm-fixture-staged-caix"
blocked_repo="redhillsmediafl/rhm-fixture-blocked-mtp-caix"
draft_repo="redhillsmediafl/rhm-fixture-draft-caix"
revision="1111111111111111111111111111111111111111"
local_dir="fixture-coreai"

manifest="$tmpdir/MANIFEST.tsv"
revisions="$tmpdir/revisions.tsv"

cat > "$manifest" <<EOF
repo	local_dir	kind	benchmark_mode	status	notes
$repo	$local_dir	standalone	decode	eligible	verified
$mtp_repo	fixture-mtp-coreai	mtp	eagle-mtp	eligible	benchmark MTP package; compare against standalone target row
$staged_repo	fixture-staged-coreai	staged	manual	component_only	staged manifest; run distributed hardware smoke on 64 GB Studio plus 32 GB MacBook
$blocked_repo	fixture-blocked-mtp-coreai	mtp	eagle-mtp	blocked_runtime	draft graph is standard two-input assistant; rebuild package with dependent EAGLE draft
$draft_repo	fixture-draft-coreai	draft	manual	component_only	draft component; benchmark with matching target
EOF

cat > "$revisions" <<EOF
$repo	$revision
$mtp_repo	$revision
$staged_repo	$revision
$blocked_repo	$revision
$draft_repo	$revision
EOF

write_raw_fixture() {
  local raw_root="$1"
  local model_dir="$raw_root/20260703-000001-fixture-coreai"
  mkdir -p "$model_dir"

  cat > "$model_dir/metadata.txt" <<EOF
name=$local_dir
model=/tmp/$local_dir
draft=
repo=$repo
repo_revision=$revision
caix_bin=./.build/release/caix
caix_commit=$(git -C "$REPO_DIR" rev-parse HEAD)
git_status=0 dirty entries
machine=test machine
memory_bytes=68719476736
os=27.0 (test)
max_tokens=8
temperature=0
seed=
warmup=0
runs=3
raw=1
draft_tokens=4
benchmark_mode=decode
prompt=tester request contract prompt
EOF

  cat > "$model_dir/summary.tsv" <<EOF
phase	run	status	generated	load_s	prefill_s	decode_s	decode_tps	stdout	stderr
measured	1	ok	8	1.0	0.1	0.5	16.0	$model_dir/measured-1.stdout.txt	$model_dir/measured-1.stderr.txt
measured	2	ok	8	1.1	0.1	0.5	16.0	$model_dir/measured-2.stdout.txt	$model_dir/measured-2.stderr.txt
measured	3	ok	8	1.2	0.1	0.5	16.0	$model_dir/measured-3.stdout.txt	$model_dir/measured-3.stderr.txt
EOF

  printf 'deterministic output\n' > "$model_dir/measured-1.stdout.txt"
  printf 'deterministic output\n' > "$model_dir/measured-2.stdout.txt"
  printf 'deterministic output\n' > "$model_dir/measured-3.stdout.txt"
  printf '[coreai] measured\n' > "$model_dir/measured-1.stderr.txt"
  printf '[coreai] measured\n' > "$model_dir/measured-2.stderr.txt"
  printf '[coreai] measured\n' > "$model_dir/measured-3.stderr.txt"
}

section_contains_repo() {
  local doc="$1"
  local start="$2"
  local stop="$3"
  awk -v start="$start" -v stop="$stop" -v repo="$repo" '
    index($0, start) == 1 { in_section = 1; next }
    index($0, stop) == 1 { in_section = 0 }
    in_section && index($0, repo) { found = 1 }
    END { exit found ? 0 : 1 }
  ' "$doc"
}

section_contains_text() {
  local doc="$1"
  local start="$2"
  local stop="$3"
  local text="$4"
  awk -v start="$start" -v stop="$stop" -v text="$text" '
    index($0, start) == 1 { in_section = 1; next }
    index($0, stop) == 1 { in_section = 0 }
    in_section && index($0, text) { found = 1 }
    END { exit found ? 0 : 1 }
  ' "$doc"
}

external_raw="$tmpdir/raw"
write_raw_fixture "$external_raw"
external_doc="$tmpdir/external.md"
"$SCRIPT_DIR/generate-tester-requests.sh" \
  --manifest "$manifest" \
  --revisions "$revisions" \
  --raw-dir "$external_raw" \
  --out "$external_doc" >/dev/null

section_contains_repo "$external_doc" "## Existing Raw Evidence" "## Manual Or Component Requests" \
  || { echo "error: valid out-of-tree raw evidence was not listed as existing evidence" >&2; exit 1; }
if section_contains_repo "$external_doc" "## Ready Benchmark Requests" "## Existing Raw Evidence"; then
  echo "error: repo with valid raw evidence was still listed as a ready tester request" >&2
  exit 1
fi
"$SCRIPT_DIR/check-tester-requests.sh" \
  --manifest "$manifest" \
  --revisions "$revisions" \
  --raw-dir "$external_raw" \
  --doc "$external_doc" >/dev/null

section_contains_text "$external_doc" "## Ready Benchmark Requests" "## Existing Raw Evidence" \
  "| \`$mtp_repo\` | \`$revision\` | \`fixture-mtp-coreai\` | EAGLE MTP load, generation, benchmark | benchmark MTP package; compare against standalone target row |" \
  || { echo "error: eligible MTP row did not keep EAGLE MTP tester request wording" >&2; exit 1; }
section_contains_text "$external_doc" "## Manual Or Component Requests" "## Run Template" \
  "| \`$staged_repo\` | \`$revision\` | \`fixture-staged-coreai\` | distributed hardware smoke | staged manifest; run distributed hardware smoke on 64 GB Studio plus 32 GB MacBook |" \
  || { echo "error: staged row did not keep distributed hardware smoke wording" >&2; exit 1; }
section_contains_text "$external_doc" "## Manual Or Component Requests" "## Run Template" \
  "| \`$blocked_repo\` | \`$revision\` | \`fixture-blocked-mtp-coreai\` | blocked; do not test | draft graph is standard two-input assistant; rebuild package with dependent EAGLE draft |" \
  || { echo "error: blocked MTP row did not keep blocked/do-not-test wording" >&2; exit 1; }
section_contains_text "$external_doc" "## Manual Or Component Requests" "## Run Template" \
  "| \`$draft_repo\` | \`$revision\` | \`fixture-draft-coreai\` | component; do not test alone | draft component; benchmark with matching target |" \
  || { echo "error: draft row did not keep component/do-not-test-alone wording" >&2; exit 1; }
rg -q 'remain `needs-test` while the second Mac is unavailable' "$external_doc" \
  || { echo "error: staged tester-request caveat must mention needs-test while the second Mac is unavailable" >&2; exit 1; }
rg -q 'not Studio-only loopback, plan dry-runs, or HF diagnostic' "$external_doc" \
  || { echo "error: staged tester-request caveat must reject Studio-only diagnostic evidence" >&2; exit 1; }

# In-repo raw evidence must be tracked before it can suppress tester requests. This prevents
# locally generated or dirty benchmark folders from silently promoting a row to existing evidence.
inside_raw="$inside_repo_tmp/raw"
write_raw_fixture "$inside_raw"
inside_doc="$tmpdir/inside-untracked.md"
"$SCRIPT_DIR/generate-tester-requests.sh" \
  --manifest "$manifest" \
  --revisions "$revisions" \
  --raw-dir "$inside_raw" \
  --out "$inside_doc" >/dev/null

section_contains_repo "$inside_doc" "## Ready Benchmark Requests" "## Existing Raw Evidence" \
  || { echo "error: untracked in-repo raw evidence should leave the repo in ready requests" >&2; exit 1; }
if section_contains_repo "$inside_doc" "## Existing Raw Evidence" "## Manual Or Component Requests"; then
  echo "error: untracked in-repo raw evidence was counted as existing evidence" >&2
  exit 1
fi

echo "tester requests contract ok"
