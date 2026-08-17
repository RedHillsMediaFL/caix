#!/usr/bin/env bash
# Self-test scripts/check-model-revisions.sh without network or real Hub state.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK="$SCRIPT_DIR/check-model-revisions.sh"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-model-revisions-contract.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

manifest="$tmpdir/MANIFEST.tsv"
revisions="$tmpdir/revisions.tsv"

cat > "$manifest" <<'EOF'
repo	local_dir	kind	benchmark_mode	status	notes
redhillsmediafl/rhm-a-caix	a-coreai	standalone	decode	eligible	ok
redhillsmediafl/rhm-b-caix	b-coreai	standalone	decode	eligible	ok
EOF

cat > "$revisions" <<'EOF'
redhillsmediafl/rhm-a-caix	1111111111111111111111111111111111111111
redhillsmediafl/rhm-b-caix	2222222222222222222222222222222222222222
EOF

"$CHECK" --manifest "$manifest" --revisions "$revisions" >/dev/null

if "$CHECK" --manifest "$manifest" --revisions "$tmpdir/missing-file.tsv" >"$tmpdir/missing-file.out" 2>&1; then
  echo "error: missing revisions-file fixture unexpectedly passed" >&2
  exit 1
fi
rg -q 'collect-model-revisions.sh --manifest' "$tmpdir/missing-file.out" || {
  echo "error: missing revisions-file failure did not include the collection command" >&2
  cat "$tmpdir/missing-file.out" >&2
  exit 1
}

cat > "$revisions" <<'EOF'
redhillsmediafl/rhm-a-caix	1111111111111111111111111111111111111111
EOF
if "$CHECK" --manifest "$manifest" --revisions "$revisions" >"$tmpdir/missing.out" 2>&1; then
  echo "error: missing revision fixture unexpectedly passed" >&2
  exit 1
fi
rg -q 'missing revision for manifest repo redhillsmediafl/rhm-b-caix' "$tmpdir/missing.out" || {
  echo "error: missing revision failure did not identify the manifest repo" >&2
  cat "$tmpdir/missing.out" >&2
  exit 1
}

cat > "$revisions" <<'EOF'
redhillsmediafl/rhm-a-caix	1111111111111111111111111111111111111111
redhillsmediafl/rhm-b-caix	2222222222222222222222222222222222222222
redhillsmediafl/rhm-stale-caix	3333333333333333333333333333333333333333
EOF
if "$CHECK" --manifest "$manifest" --revisions "$revisions" >"$tmpdir/stale.out" 2>&1; then
  echo "error: stale revision fixture unexpectedly passed" >&2
  exit 1
fi
rg -q 'stale revision for repo not in manifest: redhillsmediafl/rhm-stale-caix' "$tmpdir/stale.out" || {
  echo "error: stale revision failure did not identify the stale repo" >&2
  cat "$tmpdir/stale.out" >&2
  exit 1
}

cat > "$revisions" <<'EOF'
redhillsmediafl/rhm-a-caix	1111111111111111111111111111111111111111
redhillsmediafl/rhm-a-caix	2222222222222222222222222222222222222222
redhillsmediafl/rhm-b-caix	3333333333333333333333333333333333333333
EOF
if "$CHECK" --manifest "$manifest" --revisions "$revisions" >"$tmpdir/duplicate.out" 2>&1; then
  echo "error: duplicate revision fixture unexpectedly passed" >&2
  exit 1
fi
rg -q 'duplicates repo redhillsmediafl/rhm-a-caix' "$tmpdir/duplicate.out" || {
  echo "error: duplicate revision failure did not identify the duplicate repo" >&2
  cat "$tmpdir/duplicate.out" >&2
  exit 1
}

cat > "$revisions" <<'EOF'
redhillsmediafl/rhm-a-caix	not-a-sha
redhillsmediafl/rhm-b-caix	2222222222222222222222222222222222222222
EOF
if "$CHECK" --manifest "$manifest" --revisions "$revisions" >"$tmpdir/bad-sha.out" 2>&1; then
  echo "error: malformed revision fixture unexpectedly passed" >&2
  exit 1
fi
rg -q 'revision for redhillsmediafl/rhm-a-caix is not a 40-character lowercase SHA' "$tmpdir/bad-sha.out" || {
  echo "error: malformed revision failure did not identify the bad SHA" >&2
  cat "$tmpdir/bad-sha.out" >&2
  exit 1
}

echo "model revisions contract ok"
