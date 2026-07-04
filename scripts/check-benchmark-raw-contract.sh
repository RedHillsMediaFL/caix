#!/usr/bin/env bash
# Self-test benchmark raw/report evidence validators with local fixtures only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-benchmark-raw-contract.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

write_fixture() {
  local raw_dir="$1"
  local suite="$raw_dir/20260703-000000-suite"
  local model="$raw_dir/20260703-000001-test-model"
  local caix_commit
  caix_commit="$(git -C "$REPO_DIR" rev-parse HEAD)"

  mkdir -p "$suite" "$model"

  cat > "$suite/metadata.txt" <<EOF
manifest=$REPO_DIR/benchmarks/MANIFEST.tsv
revisions=benchmarks/revisions.tsv
exports=$REPO_DIR/models/exports
caix_commit=$caix_commit
git_status=0 dirty entries
machine=test machine
memory_bytes=68719476736
os=27.0 (test)
max_tokens=8
temperature=0
repo_revision=unknown
warmup=0
runs=2
raw=1
dry_run=0
prompt=benchmark contract prompt
total=1
measured=1
planned=0
skipped=0
failed=0
EOF

  cat > "$suite/summary.tsv" <<EOF
repo	local_dir	kind	benchmark_mode	status	reason	bundle	output
redhillsmediafl/rhm-test-caix	test-model	language	decode	measured	-	/tmp/test-model	$model
EOF

  cat > "$model/metadata.txt" <<EOF
name=test-model
model=/tmp/test-model
draft=
repo=redhillsmediafl/rhm-test-caix
repo_revision=1111111111111111111111111111111111111111
caix_bin=./.build/release/caix
caix_commit=$caix_commit
git_status=0 dirty entries
machine=test machine
memory_bytes=68719476736
os=27.0 (test)
max_tokens=8
temperature=0
seed=
warmup=0
runs=2
raw=1
draft_tokens=4
benchmark_mode=decode
prompt=benchmark contract prompt
EOF

  cat > "$model/summary.tsv" <<EOF
phase	run	status	generated	load_s	prefill_s	decode_s	decode_tps	stdout	stderr
measured	1	ok	8	1.0	0.1	0.5	16.0	$model/measured-1.stdout.txt	$model/measured-1.stderr.txt
measured	2	ok	8	1.1	0.1	0.5	16.0	$model/measured-2.stdout.txt	$model/measured-2.stderr.txt
EOF

  printf 'deterministic output\n' > "$model/measured-1.stdout.txt"
  printf 'deterministic output\n' > "$model/measured-2.stdout.txt"
  printf '[coreai] 8 generated, load=1.0s prefill=0.1s decode=0.5s (16.0 tok/s)\n' > "$model/measured-1.stderr.txt"
  printf '[coreai] 8 generated, load=1.1s prefill=0.1s decode=0.5s (16.0 tok/s)\n' > "$model/measured-2.stderr.txt"
}

valid_raw="$tmpdir/valid/raw"
write_fixture "$valid_raw"

"$SCRIPT_DIR/check-benchmark-raw.sh" --raw-dir "$valid_raw" >/dev/null
"$SCRIPT_DIR/benchmark-report.sh" \
  --suite "$valid_raw/20260703-000000-suite" \
  --out "$tmpdir/valid-report.tsv" >/dev/null

bad_suite_raw="$tmpdir/bad-suite/raw"
write_fixture "$bad_suite_raw"
python3 - "$bad_suite_raw/20260703-000000-suite/summary.tsv" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = path.read_text().splitlines()
lines[0] = "repo\tlocal_dir\tkind\tstatus\treason\tbundle\toutput"
path.write_text("\n".join(lines) + "\n")
PY

if "$SCRIPT_DIR/check-benchmark-raw.sh" --raw-dir "$bad_suite_raw" >"$tmpdir/bad-suite.out" 2>&1; then
  echo "error: stale suite summary schema unexpectedly passed benchmark raw validation" >&2
  exit 1
fi
grep -F 'suite summary schema is stale' "$tmpdir/bad-suite.out" >/dev/null \
  || { echo "error: stale suite summary failure did not mention schema" >&2; cat "$tmpdir/bad-suite.out" >&2; exit 1; }

bad_model_raw="$tmpdir/bad-model/raw"
write_fixture "$bad_model_raw"
python3 - "$bad_model_raw/20260703-000001-test-model/summary.tsv" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = path.read_text().splitlines()
lines[0] = "phase\trun\tstatus\tgenerated\tload_s\tprefill_s\tdecode_s\tstdout\tstderr"
path.write_text("\n".join(lines) + "\n")
PY

if "$SCRIPT_DIR/check-benchmark-raw.sh" --raw-dir "$bad_model_raw" >"$tmpdir/bad-model.out" 2>&1; then
  echo "error: stale model summary schema unexpectedly passed benchmark raw validation" >&2
  exit 1
fi
grep -F 'raw summary schema is stale' "$tmpdir/bad-model.out" >/dev/null \
  || { echo "error: stale model summary failure did not mention schema" >&2; cat "$tmpdir/bad-model.out" >&2; exit 1; }

if "$SCRIPT_DIR/benchmark-report.sh" \
    --suite "$bad_suite_raw/20260703-000000-suite" \
    --out "$tmpdir/bad-suite-report.tsv" >"$tmpdir/bad-suite-report.out" 2>&1
then
  echo "error: stale suite summary schema unexpectedly passed benchmark-report" >&2
  exit 1
fi
grep -F 'suite summary schema is stale' "$tmpdir/bad-suite-report.out" >/dev/null \
  || { echo "error: stale benchmark-report failure did not mention suite schema" >&2; cat "$tmpdir/bad-suite-report.out" >&2; exit 1; }

echo "benchmark raw contract ok"
