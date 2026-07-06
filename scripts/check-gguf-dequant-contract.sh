#!/usr/bin/env bash
# Regression contract for GGUF dequant model-file selection. Keeps mmproj sidecars
# out of the text-model dequant path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

python3 - "$REPO_DIR/python/converter/gguf_dequant.py" <<'PY'
import contextlib
import importlib.util
import io
import json
import sys
import types

gguf_dequant_path = sys.argv[1]
spec = importlib.util.spec_from_file_location("gguf_dequant_under_test", gguf_dequant_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

files = [
    "mmproj-F32.gguf",
    "mmproj-F16.gguf",
    "mmproj-BF16.gguf",
    "model-IQ4_XS.GGUF",
    "model-Q8_0.gguf",
]

assert module.is_mmproj_gguf("mmproj-F16.gguf") is True
assert module.is_mmproj_gguf("nested/mmproj-BF16.gguf") is True
assert module.is_mmproj_gguf("model-F16.gguf") is False
assert module.pick_gguf(files) == "model-Q8_0.gguf"
assert module.pick_gguf(["mmproj-F16.GGUF", "mmproj-BF16.gguf"]) is None

fake_transformers = types.ModuleType("transformers")
fake_transformers.AutoModelForCausalLM = object()
fake_transformers.AutoTokenizer = object()
sys.modules["transformers"] = fake_transformers


class FakeHfApi:
    files = []

    def list_repo_files(self, repo_id):
        assert repo_id == "example/gguf"
        return list(self.files)


fake_hub = types.ModuleType("huggingface_hub")
fake_hub.HfApi = FakeHfApi
sys.modules["huggingface_hub"] = fake_hub

old_argv = sys.argv
sys.argv = [
    "gguf_dequant.py",
    "--repo", "example/gguf",
    "--gguf-file", "mmproj-F16.gguf",
    "--out", "/tmp/out",
]
stdout = io.StringIO()
try:
    with contextlib.redirect_stdout(stdout):
        code = module.main()
finally:
    sys.argv = old_argv

assert code == 1
payload = json.loads(stdout.getvalue())
assert payload["ok"] is False
assert "sidecars" in payload["reason"]

FakeHfApi.files = ["mmproj-F32.gguf", "mmproj-F16.GGUF"]
old_argv = sys.argv
sys.argv = [
    "gguf_dequant.py",
    "--repo", "example/gguf",
    "--out", "/tmp/out",
]
stdout = io.StringIO()
try:
    with contextlib.redirect_stdout(stdout):
        code = module.main()
finally:
    sys.argv = old_argv

assert code == 1
payload = json.loads(stdout.getvalue())
assert payload["ok"] is False
assert "only mmproj GGUF sidecars" in payload["reason"]

print("gguf dequant contract ok")
PY
