#!/usr/bin/env bash
# Self-test public-copy guardrails with local fixtures only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-public-copy-contract.XXXXXX")"
trap 'rm -rf "$tmpdir"' EXIT

allowed="$tmpdir/allowed.md"
blocked="$tmpdir/blocked.md"

cat > "$allowed" <<'EOF'
# Local Review Notes

Structured output requires a real model smoke before public claims.
The response_format path is unsupported until a backend constrained-decoding gate passes.
Exact continuation reuse may apply in loaded CoreAILM fast handles.
Partial-prefix reuse is unsupported until a release gate exists.
Multimodal requests are parsed but not supported by any verified runtime bundle yet.
Image, audio, and video requests return HTTP 400 until runtime serving evidence exists.
Staged bundles are distributed artifacts; same-machine staged is not a single-device fast path.
The monolithic 4-bit path is deterministic 4-bit greedy and must not claim fp16 1:1.
Internal directional speed evidence: monolithic `171.3` tok/s vs staged `38.2` tok/s; do not publish.
Do not claim staged upload readiness until two-machine hardware evidence and sign-off exist.
Before asking for MacBook Thunderbolt testing, the readiness gate must pass.
RDMA over Thunderbolt 5 is a future hardware-gated transport design, not hardware evidence.
Do not claim RDMA support without a TB5 pair, captured negotiation evidence, and sign-off.
EOF

"$SCRIPT_DIR/check-public-copy.sh" "$allowed" >/dev/null

cat > "$blocked" <<'EOF'
# Release Notes

Structured output is supported for local models.
EOF

if "$SCRIPT_DIR/check-public-copy.sh" "$blocked" >"$tmpdir/blocked.out" 2>&1; then
  echo "error: structured-output support claim unexpectedly passed" >&2
  exit 1
fi
rg -q 'structured-output public claim requires real model smoke evidence' "$tmpdir/blocked.out" || {
  echo "error: structured-output failure did not mention smoke evidence" >&2
  cat "$tmpdir/blocked.out" >&2
  exit 1
}

cat > "$blocked" <<'EOF'
# Release Notes

Prompt caching is supported for all requests.
EOF

if "$SCRIPT_DIR/check-public-copy.sh" "$blocked" >"$tmpdir/prefix-cache.out" 2>&1; then
  echo "error: broad prefix-cache support claim unexpectedly passed" >&2
  exit 1
fi
rg -q 'prefix-cache public claim must be limited' "$tmpdir/prefix-cache.out" || {
  echo "error: prefix-cache failure did not mention exact continuation limit" >&2
  cat "$tmpdir/prefix-cache.out" >&2
  exit 1
}

cat > "$blocked" <<'EOF'
# Release Notes

Semantic prefix reuse and SSD-persistent KV snapshots are available.
EOF

if "$SCRIPT_DIR/check-public-copy.sh" "$blocked" >"$tmpdir/prefix-cache-advanced.out" 2>&1; then
  echo "error: advanced prefix-cache support claim unexpectedly passed" >&2
  exit 1
fi
rg -q 'prefix-cache public claim must be limited' "$tmpdir/prefix-cache-advanced.out" || {
  echo "error: advanced prefix-cache failure did not mention exact continuation limit" >&2
  cat "$tmpdir/prefix-cache-advanced.out" >&2
  exit 1
}

cat > "$blocked" <<'EOF'
# Release Notes

Multimodal image input is supported for local models.
EOF

if "$SCRIPT_DIR/check-public-copy.sh" "$blocked" >"$tmpdir/multimodal.out" 2>&1; then
  echo "error: multimodal support claim unexpectedly passed" >&2
  exit 1
fi
rg -q 'multimodal public claim requires a verified runtime bundle' "$tmpdir/multimodal.out" || {
  echo "error: multimodal failure did not mention runtime bundle evidence" >&2
  cat "$tmpdir/multimodal.out" >&2
  exit 1
}

cat > "$blocked" <<'EOF'
# Release Notes

Vision requests are verified in caix serving.
EOF

if "$SCRIPT_DIR/check-public-copy.sh" "$blocked" >"$tmpdir/vision.out" 2>&1; then
  echo "error: vision serving claim unexpectedly passed" >&2
  exit 1
fi
rg -q 'multimodal public claim requires a verified runtime bundle' "$tmpdir/vision.out" || {
  echo "error: vision failure did not mention runtime bundle evidence" >&2
  cat "$tmpdir/vision.out" >&2
  exit 1
}

cat > "$blocked" <<'EOF'
# Release Notes

The staged bundle is the single-device fast path for local serving.
EOF

if "$SCRIPT_DIR/check-public-copy.sh" "$blocked" >"$tmpdir/staged-fast.out" 2>&1; then
  echo "error: staged single-device fast-path claim unexpectedly passed" >&2
  exit 1
fi
rg -q 'serving-path public copy must keep monolithic=single-device fast' "$tmpdir/staged-fast.out" || {
  echo "error: staged-fast failure did not mention serving-path labels" >&2
  cat "$tmpdir/staged-fast.out" >&2
  exit 1
}

cat > "$blocked" <<'EOF'
# Release Notes

The 4-bit monolithic bundle is fp16-1:1 with HF.
EOF

if "$SCRIPT_DIR/check-public-copy.sh" "$blocked" >"$tmpdir/fp16-claim.out" 2>&1; then
  echo "error: 4-bit fp16 1:1 claim unexpectedly passed" >&2
  exit 1
fi
rg -q '4-bit bundles must not claim fp16 1:1' "$tmpdir/fp16-claim.out" || {
  echo "error: fp16-claim failure did not mention 4-bit/fp16 label" >&2
  cat "$tmpdir/fp16-claim.out" >&2
  exit 1
}

cat > "$blocked" <<'EOF'
# Release Notes

The staged distributed package is ready for Thunderbolt testing.
EOF

if "$SCRIPT_DIR/check-public-copy.sh" "$blocked" >"$tmpdir/distributed-ready.out" 2>&1; then
  echo "error: distributed ready-to-test claim unexpectedly passed" >&2
  exit 1
fi
rg -q 'distributed readiness/upload claims require two-machine hardware evidence' "$tmpdir/distributed-ready.out" || {
  echo "error: distributed-ready failure did not mention hardware evidence" >&2
  cat "$tmpdir/distributed-ready.out" >&2
  exit 1
}

cat > "$blocked" <<'EOF'
# Release Notes

The staged artifact is upload-ready after the Studio-only parity run.
EOF

if "$SCRIPT_DIR/check-public-copy.sh" "$blocked" >"$tmpdir/staged-upload-ready.out" 2>&1; then
  echo "error: staged upload-ready claim unexpectedly passed" >&2
  exit 1
fi
rg -q 'distributed readiness/upload claims require two-machine hardware evidence' "$tmpdir/staged-upload-ready.out" || {
  echo "error: staged-upload failure did not mention hardware evidence" >&2
  cat "$tmpdir/staged-upload-ready.out" >&2
  exit 1
}

cat > "$blocked" <<'EOF'
# Release Notes

RDMA over Thunderbolt 5 is supported for distributed serving.
EOF

if "$SCRIPT_DIR/check-public-copy.sh" "$blocked" >"$tmpdir/rdma-supported.out" 2>&1; then
  echo "error: RDMA support claim unexpectedly passed" >&2
  exit 1
fi
rg -q 'RDMA/TB5 public claims require TB5 hardware evidence' "$tmpdir/rdma-supported.out" || {
  echo "error: RDMA support failure did not mention TB5 hardware evidence" >&2
  cat "$tmpdir/rdma-supported.out" >&2
  exit 1
}

cat > "$blocked" <<'EOF'
# Release Notes

The TB5 transport is production-ready for distributed inference.
EOF

if "$SCRIPT_DIR/check-public-copy.sh" "$blocked" >"$tmpdir/tb5-production.out" 2>&1; then
  echo "error: TB5 production-ready claim unexpectedly passed" >&2
  exit 1
fi
rg -q 'RDMA/TB5 public claims require TB5 hardware evidence' "$tmpdir/tb5-production.out" || {
  echo "error: TB5 production failure did not mention TB5 hardware evidence" >&2
  cat "$tmpdir/tb5-production.out" >&2
  exit 1
}

cat > "$blocked" <<'EOF'
# Benchmark Notes

This model decodes at 171.3 tok/s in local testing.
EOF

if "$SCRIPT_DIR/check-public-copy.sh" "$blocked" >"$tmpdir/bench.out" 2>&1; then
  echo "error: raw public speed number unexpectedly passed" >&2
  exit 1
fi
rg -q 'raw benchmark speed number in public copy' "$tmpdir/bench.out" || {
  echo "error: raw-speed failure did not mention public speed copy" >&2
  cat "$tmpdir/bench.out" >&2
  exit 1
}

cat > "$blocked" <<'EOF'
# Benchmark Notes

The local path is `171.3` tok/s and 4.5x faster.
EOF

if "$SCRIPT_DIR/check-public-copy.sh" "$blocked" >"$tmpdir/bench-markdown.out" 2>&1; then
  echo "error: markdown speed claim unexpectedly passed" >&2
  exit 1
fi
rg -q 'public speed claims require publishable raw benchmark evidence' "$tmpdir/bench-markdown.out" || {
  echo "error: markdown-speed failure did not mention publishable raw evidence" >&2
  cat "$tmpdir/bench-markdown.out" >&2
  exit 1
}

echo "public-copy contract ok"
