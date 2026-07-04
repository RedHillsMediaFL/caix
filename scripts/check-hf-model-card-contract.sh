#!/usr/bin/env bash
# Self-test the Hugging Face model-card contract using local fixtures only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-hf-card-contract.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

manifest="$tmpdir/MANIFEST.tsv"
cards_dir="$tmpdir/cards"
mkdir -p "$cards_dir"

cat > "$manifest" <<'TSV'
repo	local_dir	kind	benchmark_mode	status	notes
redhillsmediafl/rhm-ready-caix	ready	standalone	decode	eligible	verified
redhillsmediafl/rhm-gated-caix	gated	standalone	decode	eligible	monolithic default <=16 prefill mitigation required; deterministic 4bit path is non-1:1 vs fp16
redhillsmediafl/rhm-staged-caix	staged	staged	manual	component_only	staged manifest; run distributed hardware smoke
redhillsmediafl/rhm-mtp-caix	mtp	mtp	eagle-mtp	eligible	benchmark MTP package; compare against standalone target row
TSV

base_card() {
  local title="$1"
  cat <<EOF
---
library_name: caix
---
# $title

| Base model | source |
| Format | caix |
| Quant | 4bit |
| Context | documented |
| Runtime | caix |
| License | source |

## Download

Use caix catalog tools.

## License

Source license applies.

Support: https://redhillsmediafl.com/open-source
EOF
}

{
  base_card "ready"
  cat <<'EOF'

This bundle is verified in caix.

Parity evidence: teacher-forced HF-fp16 parity gate.
Speed evidence: raw benchmark logs.
EOF
} > "$cards_dir/redhillsmediafl__rhm-ready-caix.README.md"

{
  base_card "gated"
  cat <<'EOF'

## caix-status-label

non-1:1/4bit-deterministic; default prefill-chunk determinism gate required before verified status.
EOF
} > "$cards_dir/redhillsmediafl__rhm-gated-caix.README.md"

{
  base_card "staged"
  cat <<'EOF'

## caix-status-label

component-only needs-test package; not a standalone target and requires distributed hardware smoke.
EOF
} > "$cards_dir/redhillsmediafl__rhm-staged-caix.README.md"

{
  base_card "mtp"
  cat <<'EOF'

## caix-status-label

MTP target+draft package; benchmark against the matching standalone target row and report MTP
numbers separately.
EOF
} > "$cards_dir/redhillsmediafl__rhm-mtp-caix.README.md"

"$SCRIPT_DIR/check-hf-model-cards.sh" --manifest "$manifest" --cards-dir "$cards_dir" >/dev/null

cp "$cards_dir/redhillsmediafl__rhm-gated-caix.README.md" "$tmpdir/gated.good"
perl -0pi -e 's/## caix-status-label\n\nnon-1:1\/4bit-deterministic; default prefill-chunk determinism gate required before verified status\.\n//' \
  "$cards_dir/redhillsmediafl__rhm-gated-caix.README.md"
if "$SCRIPT_DIR/check-hf-model-cards.sh" --manifest "$manifest" --cards-dir "$cards_dir" >"$tmpdir/gated.out" 2>&1; then
  echo "error: missing caix-status-label fixture unexpectedly passed" >&2
  exit 1
fi
rg -q 'caix-status-label' "$tmpdir/gated.out" || {
  echo "error: missing-label failure did not mention caix-status-label" >&2
  cat "$tmpdir/gated.out" >&2
  exit 1
}
cp "$tmpdir/gated.good" "$cards_dir/redhillsmediafl__rhm-gated-caix.README.md"

cp "$cards_dir/redhillsmediafl__rhm-mtp-caix.README.md" "$tmpdir/mtp.good"
perl -0pi -e 's/## caix-status-label\n\nMTP target\+draft package; benchmark against the matching standalone target row and report MTP\nnumbers separately\.\n//' \
  "$cards_dir/redhillsmediafl__rhm-mtp-caix.README.md"
if "$SCRIPT_DIR/check-hf-model-cards.sh" --manifest "$manifest" --cards-dir "$cards_dir" >"$tmpdir/mtp.out" 2>&1; then
  echo "error: MTP card without caix-status-label unexpectedly passed" >&2
  exit 1
fi
rg -q 'MTP/speculative card must include a caix-status-label block' "$tmpdir/mtp.out" || {
  echo "error: missing-MTP-label failure did not mention caix-status-label" >&2
  cat "$tmpdir/mtp.out" >&2
  exit 1
}
cp "$tmpdir/mtp.good" "$cards_dir/redhillsmediafl__rhm-mtp-caix.README.md"

cp "$cards_dir/redhillsmediafl__rhm-ready-caix.README.md" "$tmpdir/ready.good"
perl -0pi -e 's/\nSpeed evidence: raw benchmark logs\.\n//' \
  "$cards_dir/redhillsmediafl__rhm-ready-caix.README.md"
if "$SCRIPT_DIR/check-hf-model-cards.sh" --manifest "$manifest" --cards-dir "$cards_dir" >"$tmpdir/ready.out" 2>&1; then
  echo "error: ready card without speed evidence unexpectedly passed" >&2
  exit 1
fi
rg -q 'speed evidence' "$tmpdir/ready.out" || {
  echo "error: missing-speed failure did not mention speed evidence" >&2
  cat "$tmpdir/ready.out" >&2
  exit 1
}
cp "$tmpdir/ready.good" "$cards_dir/redhillsmediafl__rhm-ready-caix.README.md"

perl -0pi -e 's/component-only needs-test package/This component is verified in caix. component-only needs-test package/' \
  "$cards_dir/redhillsmediafl__rhm-staged-caix.README.md"
if "$SCRIPT_DIR/check-hf-model-cards.sh" --manifest "$manifest" --cards-dir "$cards_dir" >"$tmpdir/staged.out" 2>&1; then
  echo "error: component card with verified claim unexpectedly passed" >&2
  exit 1
fi
rg -q 'must not claim ready-to-run or verified status' "$tmpdir/staged.out" || {
  echo "error: component-ready failure did not mention forbidden verified status" >&2
  cat "$tmpdir/staged.out" >&2
  exit 1
}

echo "hf model-card contract ok"
