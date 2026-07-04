#!/usr/bin/env bash
# Self-test benchmark-gap strict/non-strict behavior with local fixtures only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-benchmark-gaps-contract.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

revision="1111111111111111111111111111111111111111"
manifest="$tmpdir/MANIFEST.tsv"
raw_dir="$tmpdir/raw"

cat > "$manifest" <<'EOF'
repo	local_dir	kind	benchmark_mode	status	notes
redhillsmediafl/rhm-measured-caix	measured-coreai	standalone	decode	eligible	verified
redhillsmediafl/rhm-pending-caix	pending-coreai	standalone	decode	eligible	needs raw evidence
redhillsmediafl/rhm-mode-split-caix	mode-split-coreai	standalone	decode	eligible	decode has evidence
redhillsmediafl/rhm-mode-split-caix	mode-split-mtp-coreai	mtp	speculative	eligible	speculative package needs separate target+draft evidence
redhillsmediafl/rhm-manual-caix	manual-staged-coreai	staged	manual	component_only	staged hardware smoke later
EOF

write_raw_fixture() {
  local repo="$1"
  local name="$2"
  local mode="$3"
  local dir="$raw_dir/20260703-000001-$name-$mode"
  mkdir -p "$dir"

  cat > "$dir/metadata.txt" <<EOF
name=$name
model=/tmp/$name
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
benchmark_mode=$mode
prompt=benchmark gaps contract prompt
EOF

  cat > "$dir/summary.tsv" <<EOF
phase	run	status	generated	load_s	prefill_s	decode_s	decode_tps	stdout	stderr
measured	1	ok	8	1.0	0.1	0.5	16.0	$dir/measured-1.stdout.txt	$dir/measured-1.stderr.txt
measured	2	ok	8	1.1	0.1	0.5	16.0	$dir/measured-2.stdout.txt	$dir/measured-2.stderr.txt
measured	3	ok	8	1.2	0.1	0.5	16.0	$dir/measured-3.stdout.txt	$dir/measured-3.stderr.txt
EOF
}

write_raw_fixture redhillsmediafl/rhm-measured-caix measured-coreai decode
write_raw_fixture redhillsmediafl/rhm-mode-split-caix mode-split-coreai decode

if ! "$SCRIPT_DIR/check-benchmark-gaps.sh" \
    --manifest "$manifest" \
    --raw-dir "$raw_dir" >"$tmpdir/non-strict.out"
then
  echo "error: non-strict benchmark gap audit should report pending rows but exit 0" >&2
  cat "$tmpdir/non-strict.out" >&2
  exit 1
fi
grep -F 'benchmark gap audit: 2/4 eligible rows have committed measured raw evidence; pending=2; noneligible=1' \
  "$tmpdir/non-strict.out" >/dev/null \
  || { echo "error: non-strict benchmark gap summary changed unexpectedly" >&2; cat "$tmpdir/non-strict.out" >&2; exit 1; }
grep -F $'redhillsmediafl/rhm-pending-caix\tpending-coreai\tdecode\tneeds raw evidence' \
  "$tmpdir/non-strict.out" >/dev/null \
  || { echo "error: pending decode row was not reported" >&2; cat "$tmpdir/non-strict.out" >&2; exit 1; }
grep -F $'redhillsmediafl/rhm-mode-split-caix\tmode-split-mtp-coreai\tspeculative\tspeculative package needs separate target+draft evidence' \
  "$tmpdir/non-strict.out" >/dev/null \
  || { echo "error: same-repo speculative row was incorrectly satisfied by decode evidence" >&2; cat "$tmpdir/non-strict.out" >&2; exit 1; }

if "$SCRIPT_DIR/check-benchmark-gaps.sh" \
    --manifest "$manifest" \
    --raw-dir "$raw_dir" \
    --strict >"$tmpdir/strict.out" 2>&1
then
  echo "error: strict benchmark gap audit unexpectedly passed with pending rows" >&2
  cat "$tmpdir/strict.out" >&2
  exit 1
fi
grep -F 'pending=2' "$tmpdir/strict.out" >/dev/null \
  || { echo "error: strict failure did not report pending rows" >&2; cat "$tmpdir/strict.out" >&2; exit 1; }

write_raw_fixture redhillsmediafl/rhm-pending-caix pending-coreai decode
write_raw_fixture redhillsmediafl/rhm-mode-split-caix mode-split-mtp-coreai speculative
"$SCRIPT_DIR/check-benchmark-gaps.sh" \
  --manifest "$manifest" \
  --raw-dir "$raw_dir" \
  --strict >"$tmpdir/strict-clean.out"
grep -F 'benchmark gap audit ok: 4/4 eligible rows have committed measured raw evidence; noneligible=1' \
  "$tmpdir/strict-clean.out" >/dev/null \
  || { echo "error: strict clean benchmark gap summary changed unexpectedly" >&2; cat "$tmpdir/strict-clean.out" >&2; exit 1; }

echo "benchmark gaps contract ok"
