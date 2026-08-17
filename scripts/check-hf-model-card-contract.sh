#!/usr/bin/env bash
# Self-test the Hugging Face production model-card contract using local fixtures only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-hf-card-contract.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

manifest="$tmpdir/MANIFEST.tsv"
cards_dir="$tmpdir/cards"
mkdir -p "$cards_dir"

cat > "$manifest" <<'TSV'
repo	local_dir	kind	benchmark_mode	status	notes
redhillsmediafl/rhm-ready-caix	ready	standalone	decode	eligible	full-native-context quantized; speed pending
redhillsmediafl/rhm-mm-caix	mm	staged	manual	component_only	staged multimodal split-cache package
redhillsmediafl/rhm-mtp-caix	mtp	mtp	eagle-mtp	eligible	benchmark MTP package; compare against standalone target row
TSV

production_card() {
  local title="$1"
  local repo="$2"
  local bundle="$3"
  local pitch="$4"
  local extra="$5"
  cat <<EOF
---
license: apache-2.0
library_name: caix
base_model: source/model
tags:
- caix
- apple-silicon
- core-ai
---

# $title

<p align="center">
  <img src="https://huggingface.co/spaces/redhillsmediafl/README/resolve/main/rhm-full-logo.png" alt="Red Hills Media" width="120">
</p>

$pitch

## Install & run

\`\`\`bash
brew install redhillsmediafl/caix/caix
caix catalog install $repo
caix serve --model $bundle
\`\`\`

$extra

## At a glance

- **Base model:** [source/model](https://huggingface.co/source/model)
- **Context:** 32,768 tokens (matches base model)
- **Quantization:** 4-bit
- **Download:** ~1 GB
- **Runs on:** Apple silicon with caix 0.2.13-beta+

## Status

Verified beta. Full native context matches the base model. No public speed claim.

## License

Derived from [source/model](https://huggingface.co/source/model); follows the base model's license (\`apache-2.0\`). Review the base model license before use.

<sub>More open-source work: [redhillsmediafl.com/open-source](https://redhillsmediafl.com/open-source).</sub>
EOF
}

production_card \
  "Ready · caix" \
  "redhillsmediafl/rhm-ready-caix" \
  "ready" \
  "Run **Ready** locally with caix." \
  "## Which version?

| Version | Download | Notes |
|---|---|---|
| **4-bit** (this repo) | ~1 GB | Quantized caix package |
| **Full weights** | ~2 GB | Regular-weight caix package |" \
  > "$cards_dir/redhillsmediafl__rhm-ready-caix.README.md"

production_card \
  "MM · caix" \
  "redhillsmediafl/rhm-mm-caix" \
  "mm" \
  "Run **MM** locally with caix." \
  "Download is ~11 GB.

## Understands images

Verified runtime bundle: send one base64 image plus text in a standard /v1/chat/completions request." \
  > "$cards_dir/redhillsmediafl__rhm-mm-caix.README.md"

production_card \
  "MTP · caix" \
  "redhillsmediafl/rhm-mtp-caix" \
  "mtp" \
  "Run **MTP** target+draft speculative decoding locally with caix." \
  "## Which version?

| Version | Download | Notes |
|---|---|---|
| **MTP target+draft** (this repo) | ~2 GB | Speculative decoding experiments |" \
  > "$cards_dir/redhillsmediafl__rhm-mtp-caix.README.md"

"$SCRIPT_DIR/check-hf-model-cards.sh" --manifest "$manifest" --cards-dir "$cards_dir" >/dev/null

cp "$cards_dir/redhillsmediafl__rhm-ready-caix.README.md" "$tmpdir/ready.good"
perl -0pi -e 's#^base_model: source/model\n##m' \
  "$cards_dir/redhillsmediafl__rhm-ready-caix.README.md"
if "$SCRIPT_DIR/check-hf-model-cards.sh" --manifest "$manifest" --cards-dir "$cards_dir" >"$tmpdir/base-model.out" 2>&1; then
  echo "error: card without base_model unexpectedly passed" >&2
  exit 1
fi
rg -q 'base_model' "$tmpdir/base-model.out" || {
  echo "error: missing-base-model failure did not mention base_model" >&2
  cat "$tmpdir/base-model.out" >&2
  exit 1
}
cp "$tmpdir/ready.good" "$cards_dir/redhillsmediafl__rhm-ready-caix.README.md"

perl -0pi -e 's/## Status\n\nVerified beta.+?No public speed claim\.\n/## Status\n\nProduction ready.\n/s' \
  "$cards_dir/redhillsmediafl__rhm-ready-caix.README.md"
if "$SCRIPT_DIR/check-hf-model-cards.sh" --manifest "$manifest" --cards-dir "$cards_dir" >"$tmpdir/status.out" 2>&1; then
  echo "error: card without recognized status unexpectedly passed" >&2
  exit 1
fi
rg -q 'verified/needs-test/component/blocked/beta' "$tmpdir/status.out" || {
  echo "error: missing-status failure did not mention recognized status wording" >&2
  cat "$tmpdir/status.out" >&2
  exit 1
}
cp "$tmpdir/ready.good" "$cards_dir/redhillsmediafl__rhm-ready-caix.README.md"

perl -0pi -e 's/width="120"/width="260"/' \
  "$cards_dir/redhillsmediafl__rhm-ready-caix.README.md"
if "$SCRIPT_DIR/check-hf-model-cards.sh" --manifest "$manifest" --cards-dir "$cards_dir" >"$tmpdir/logo.out" 2>&1; then
  echo "error: oversized logo unexpectedly passed" >&2
  exit 1
fi
rg -q 'small RHM logo' "$tmpdir/logo.out" || {
  echo "error: oversized-logo failure did not mention the small logo" >&2
  cat "$tmpdir/logo.out" >&2
  exit 1
}
cp "$tmpdir/ready.good" "$cards_dir/redhillsmediafl__rhm-ready-caix.README.md"

perl -0pi -e 's/tags:\n- caix/tags:\n- caix\n- caix/' \
  "$cards_dir/redhillsmediafl__rhm-ready-caix.README.md"
if "$SCRIPT_DIR/check-hf-model-cards.sh" --manifest "$manifest" --cards-dir "$cards_dir" >"$tmpdir/duplicate-tags.out" 2>&1; then
  echo "error: card with duplicate tags unexpectedly passed" >&2
  exit 1
fi
rg -q 'duplicate front-matter tag' "$tmpdir/duplicate-tags.out" || {
  echo "error: duplicate-tag failure did not mention duplicate tags" >&2
  cat "$tmpdir/duplicate-tags.out" >&2
  exit 1
}
cp "$tmpdir/ready.good" "$cards_dir/redhillsmediafl__rhm-ready-caix.README.md"

perl -0pi -e 's/No public speed claim\./No public speed claim.\n\nThis none; regular bf16 full-weight export./' \
  "$cards_dir/redhillsmediafl__rhm-ready-caix.README.md"
if "$SCRIPT_DIR/check-hf-model-cards.sh" --manifest "$manifest" --cards-dir "$cards_dir" >"$tmpdir/this-none.out" 2>&1; then
  echo "error: card with unfilled compression fragment unexpectedly passed" >&2
  exit 1
fi
rg -q 'unfilled compression fragment|This none' "$tmpdir/this-none.out" || {
  echo "error: unfilled-compression failure did not mention the fragment" >&2
  cat "$tmpdir/this-none.out" >&2
  exit 1
}
cp "$tmpdir/ready.good" "$cards_dir/redhillsmediafl__rhm-ready-caix.README.md"

perl -0pi -e 's/No public speed claim\./No public speed claim.\n\nTested on a Mac Studio with Xcode beta./' \
  "$cards_dir/redhillsmediafl__rhm-ready-caix.README.md"
if "$SCRIPT_DIR/check-hf-model-cards.sh" --manifest "$manifest" --cards-dir "$cards_dir" >"$tmpdir/test-device.out" 2>&1; then
  echo "error: card with test-device details unexpectedly passed" >&2
  exit 1
fi
rg -q 'build/test device details' "$tmpdir/test-device.out" || {
  echo "error: test-device failure did not mention build/test device details" >&2
  cat "$tmpdir/test-device.out" >&2
  exit 1
}
cp "$tmpdir/ready.good" "$cards_dir/redhillsmediafl__rhm-ready-caix.README.md"

perl -0pi -e 's#<sub>More open-source work:.+?</sub>\n##s' \
  "$cards_dir/redhillsmediafl__rhm-ready-caix.README.md"
if "$SCRIPT_DIR/check-hf-model-cards.sh" --manifest "$manifest" --cards-dir "$cards_dir" >"$tmpdir/footer.out" 2>&1; then
  echo "error: card without support footer unexpectedly passed" >&2
  exit 1
fi
rg -q 'support link missing' "$tmpdir/footer.out" || {
  echo "error: missing-footer failure did not mention support link" >&2
  cat "$tmpdir/footer.out" >&2
  exit 1
}

echo "hf model-card contract ok"
