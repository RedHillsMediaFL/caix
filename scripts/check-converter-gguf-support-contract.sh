#!/usr/bin/env bash
# Regression contract for GGUF-only support checks. This avoids network access by
# injecting a fake huggingface_hub module and forcing config.json lookup to fail.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

python3 - "$REPO_DIR/python/converter/check_support.py" <<'PY'
import contextlib
import importlib.util
import io
import json
import sys
import types

check_support_path = sys.argv[1]
spec = importlib.util.spec_from_file_location("check_support_under_test", check_support_path)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


class FakeHfApi:
    files = []
    fail = False

    def __init__(self, token=None):
        self.token = token

    def list_repo_files(self, repo_id):
        assert repo_id == "example/gguf-only"
        if self.fail:
            raise RuntimeError("hub unavailable")
        return list(self.files)


fake_hub = types.ModuleType("huggingface_hub")
fake_hub.HfApi = FakeHfApi
sys.modules["huggingface_hub"] = fake_hub
module._load_config = lambda hf_id: (None, "missing config")


def run(files, *, hub_fails=False, stdlib_files=None):
    FakeHfApi.files = files
    FakeHfApi.fail = hub_fails
    if stdlib_files is not None:
        module._list_repo_files_stdlib = lambda hf_id: list(stdlib_files)
    old_argv = sys.argv
    sys.argv = ["check_support.py", "--hf-id", "example/gguf-only"]
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            code = module.main()
    finally:
        sys.argv = old_argv
        FakeHfApi.fail = False
    assert code == 0
    return json.loads(stdout.getvalue())


gguf_only = run([
    "model-Q8_0.GGUF",
    "model-Q4_K_M.gguf",
    "mmproj-F16.GGUF",
    "README.md",
])
assert gguf_only["ok"] is False
assert gguf_only["supported"] is False
assert gguf_only["gguf_only"] is True
assert gguf_only["gguf_model_file_count"] == 2
assert gguf_only["gguf_mmproj"] is True
assert gguf_only["gguf_mmproj_file_count"] == 1
assert "2 model .gguf files" in gguf_only["reason"]
assert "1 mmproj sidecar file" in gguf_only["reason"]
assert "dequantize + convert" in gguf_only["reason"]
assert "does not consume" in gguf_only["reason"]
assert "image-text conversion" in gguf_only["next_step"]

stdlib_gguf_only = run([], hub_fails=True, stdlib_files=[
    "model-Q8_0.gguf",
    "mmproj-F16.gguf",
    "README.md",
])
assert stdlib_gguf_only["gguf_only"] is True
assert stdlib_gguf_only["gguf_model_file_count"] == 1
assert stdlib_gguf_only["gguf_mmproj"] is True

mixed = run([
    "model-Q8_0.gguf",
    "model.SAFETENSORS",
])
assert mixed["ok"] is False
assert mixed["supported"] is False
assert "gguf_only" not in mixed
assert mixed["reason"] == "missing config"

with open(sys.argv[1].replace("python/converter/check_support.py", "web/index.html"), "r", encoding="utf-8") as f:
    index_html = f.read()
assert "mmproj sidecar files are not consumed yet" in index_html

with open(sys.argv[1].replace("python/converter/check_support.py", "web/chat.html"), "r", encoding="utf-8") as f:
    chat_html = f.read()
assert "mmproj sidecar files are not consumed yet" in chat_html

print("converter gguf support contract ok")
PY
