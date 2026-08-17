#!/usr/bin/env bash
# Self-test converter temp-routing invariants without running conversion.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONVERT="$REPO_DIR/python/converter/convert.py"

fail() {
  echo "error: $*" >&2
  exit 1
}

[[ -f "$CONVERT" ]] || fail "converter wrapper not found: $CONVERT"

grep -F -- 'TMPDIR_EXPORT = caix_env("caix_tmpdir", "TMPDIR", str(PIPELINE_ROOT / ".tmp" / "coreai-tmp"))' "$CONVERT" >/dev/null \
  || fail "converter must default conversion temp to .tmp/coreai-tmp via caix_tmpdir/CAIX_TMPDIR"
tmp_env_count="$(grep -F -c -- 'env = {**os.environ, "HF_HOME": HF_HOME, "TMPDIR": TMPDIR_EXPORT}' "$CONVERT")"
[[ "$tmp_env_count" -ge 2 ]] \
  || fail "GGUF dequant and export subprocesses must both receive TMPDIR_EXPORT"
grep -F -- 'tempfile.mkdtemp(prefix="caix-gguf-hf-",' "$CONVERT" >/dev/null \
  || fail "GGUF dequant staging must use an explicit temp directory"
grep -F -- 'dir=(TMPDIR_EXPORT)' "$CONVERT" >/dev/null \
  || fail "GGUF dequant staging must allocate under TMPDIR_EXPORT"

echo "converter temp contract ok"
