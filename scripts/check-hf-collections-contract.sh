#!/usr/bin/env bash
# Self-test Hugging Face collection coverage/note guardrails with local fixtures only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-hf-collections-contract.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

manifest="$tmpdir/MANIFEST.tsv"
items="$tmpdir/items.tsv"

cat > "$manifest" <<'TSV'
repo	local_dir	kind	benchmark_mode	status	notes
redhillsmediafl/rhm-ready-caix	ready	standalone	decode	eligible	verified
redhillsmediafl/rhm-staged-caix	staged	staged	manual	component_only	staged manifest; run distributed hardware smoke
TSV

cat > "$items" <<'TSV'
redhillsmediafl/qwen-caix-000	redhillsmediafl/rhm-ready-caix	Verified model card; see raw evidence.
redhillsmediafl/gemma-caix-000	redhillsmediafl/rhm-staged-caix	needs-test; staged manifest requires distributed hardware smoke.
TSV

"$SCRIPT_DIR/check-hf-collections.sh" --manifest "$manifest" --items-file "$items" >/dev/null

cat > "$items" <<'TSV'
redhillsmediafl/qwen-caix-000	redhillsmediafl/rhm-ready-caix	Verified model card; see raw evidence.
TSV

if "$SCRIPT_DIR/check-hf-collections.sh" --manifest "$manifest" --items-file "$items" >"$tmpdir/missing.out" 2>&1; then
  echo "error: missing collection item fixture unexpectedly passed" >&2
  exit 1
fi
rg -q 'manifest repos missing from live caix collections' "$tmpdir/missing.out" || {
  echo "error: missing-item failure did not mention collection coverage" >&2
  cat "$tmpdir/missing.out" >&2
  exit 1
}

cat > "$items" <<'TSV'
redhillsmediafl/qwen-caix-000	redhillsmediafl/rhm-ready-caix	Verified model card; see raw evidence.
redhillsmediafl/gemma-caix-000	redhillsmediafl/rhm-staged-caix	Runtime smoke pending.
TSV

if "$SCRIPT_DIR/check-hf-collections.sh" --manifest "$manifest" --items-file "$items" >"$tmpdir/pending.out" 2>&1; then
  echo "error: pending collection note fixture unexpectedly passed" >&2
  exit 1
fi
rg -q 'public collection notes need cleanup' "$tmpdir/pending.out" || {
  echo "error: pending-note failure did not mention public note cleanup" >&2
  cat "$tmpdir/pending.out" >&2
  exit 1
}

cat > "$items" <<'TSV'
redhillsmediafl/qwen-caix-000	redhillsmediafl/rhm-ready-caix	Verified model card; see raw evidence.
redhillsmediafl/gemma-caix-000	redhillsmediafl/rhm-staged-caix	Needs-test; decodes at 171.3 tok/s.
TSV

if "$SCRIPT_DIR/check-hf-collections.sh" --manifest "$manifest" --items-file "$items" >"$tmpdir/speed.out" 2>&1; then
  echo "error: speed collection note fixture unexpectedly passed" >&2
  exit 1
fi
rg -q 'public collection notes need cleanup' "$tmpdir/speed.out" || {
  echo "error: speed-note failure did not mention public note cleanup" >&2
  cat "$tmpdir/speed.out" >&2
  exit 1
}

echo "hf collection contract ok"
