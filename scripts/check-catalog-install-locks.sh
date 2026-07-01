#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAIX_BIN="${CAIX_BIN:-$ROOT/.build/debug/caix}"

if [[ ! -x "$CAIX_BIN" ]]; then
  swift build --product caix >/dev/null
fi

TMP="$(mktemp -d "${TMPDIR:-/tmp}/caix-catalog-locks.XXXXXX")"
cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  rm -rf "$TMP"
}
trap cleanup EXIT

PORT_FILE="$TMP/port"
cat >"$TMP/hub.py" <<'PY'
import http.server
import json
import pathlib
import socketserver
import sys
import urllib.parse

port_file = pathlib.Path(sys.argv[1])

class Handler(http.server.BaseHTTPRequestHandler):
    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path.startswith("/api/collections/"):
            self.send_json({"error": "not found"}, status=404)
            return
        if parsed.path == "/api/models":
            self.send_json([
                {
                    "id": "redhillsmediafl/rhm-test-caix",
                    "tags": ["caix", "base_model:test/tiny", "license:apache-2.0"],
                    "library_name": "caix",
                }
            ])
            return
        if parsed.path == "/api/models/redhillsmediafl/rhm-test-caix":
            self.send_json({
                "id": "redhillsmediafl/rhm-test-caix",
                "sha": "abc123",
                "tags": ["caix", "base_model:test/tiny", "license:apache-2.0"],
                "library_name": "caix",
                "usedStorage": 128,
                "siblings": [{"rfilename": "metadata.json"}, {"rfilename": "payload.bin"}],
            })
            return
        if parsed.path == "/redhillsmediafl/rhm-test-caix/resolve/abc123/metadata.json":
            self.send_json({
                "metadata_version": "0.2",
                "kind": "llm",
                "name": "test-coreai",
                "assets": {"main": "payload.bin"},
            })
            return
        if parsed.path == "/redhillsmediafl/rhm-test-caix/resolve/abc123/README.md":
            body = b"verified\n"
            self.send_response(200)
            self.send_header("content-type", "text/markdown")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_json({"error": "not found", "path": parsed.path}, status=404)

    def log_message(self, *_args):
        pass

with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
    port_file.write_text(str(httpd.server_address[1]))
    httpd.serve_forever()
PY

python3 "$TMP/hub.py" "$PORT_FILE" &
SERVER_PID="$!"
for _ in {1..100}; do
  [[ -s "$PORT_FILE" ]] && break
  sleep 0.05
done
[[ -s "$PORT_FILE" ]] || {
  echo "error: local test hub did not start" >&2
  exit 1
}

mkdir -p "$TMP/bin"
HF_ARGV_LOG="$TMP/hf-argv.txt"
EXPORTS="$TMP/exports"
FINAL="$EXPORTS/test-coreai"
mkdir -p "$FINAL/.cache/huggingface"
: >"$FINAL/.cache/huggingface/.gitignore.lock"
printf 'old\n' >"$FINAL/old.txt"

cat >"$TMP/bin/hf" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" >"$HF_ARGV_LOG"
local_dir=""
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --local-dir)
      shift
      local_dir="${1:?}"
      ;;
  esac
  shift || true
done
if [[ -z "$local_dir" ]]; then
  echo "missing --local-dir" >&2
  exit 2
fi
if [[ "$local_dir" == "$FINAL" ]]; then
  echo "hf was pointed at final destination" >&2
  exit 3
fi
mkdir -p "$local_dir/.cache/huggingface"
printf '{"metadata_version":"0.2","kind":"llm","name":"test-coreai","assets":{"main":"payload.bin"}}\n' >"$local_dir/metadata.json"
printf 'payload\n' >"$local_dir/payload.bin"
: >"$local_dir/.cache/huggingface/.gitignore.lock"
SH
chmod +x "$TMP/bin/hf"

PATH="$TMP/bin:$PATH" \
HF_ENDPOINT="http://127.0.0.1:$(cat "$PORT_FILE")" \
HF_ARGV_LOG="$HF_ARGV_LOG" \
FINAL="$FINAL" \
"$CAIX_BIN" catalog install redhillsmediafl/rhm-test-caix --exports "$EXPORTS" >"$TMP/install.out"

[[ -f "$FINAL/metadata.json" ]] || {
  echo "error: final metadata missing" >&2
  exit 1
}
[[ ! -e "$FINAL/old.txt" ]] || {
  echo "error: old destination was not replaced" >&2
  exit 1
}
[[ ! -e "$FINAL/.cache/huggingface/.gitignore.lock" ]] || {
  echo "error: stale Hugging Face lock leaked into final destination" >&2
  exit 1
}
[[ ! -e "$FINAL/.caix-install.lock" ]] || {
  echo "error: legacy destination install lock leaked into final destination" >&2
  exit 1
}
[[ ! -e "$EXPORTS/.test-coreai.caix-install.lock" ]] || {
  echo "error: sibling install lock was not released" >&2
  exit 1
}
if [[ -d "$EXPORTS/.caix-install-staging" ]] &&
    find "$EXPORTS/.caix-install-staging" -mindepth 1 -print -quit | read -r _; then
  echo "error: staging directory was not cleaned" >&2
  exit 1
fi
if ! grep -F -- "--local-dir" "$HF_ARGV_LOG" >/dev/null; then
  echo "error: hf was not invoked with --local-dir" >&2
  exit 1
fi
if grep -F -- "$FINAL" "$HF_ARGV_LOG" >/dev/null; then
  echo "error: hf argv used final destination instead of staging" >&2
  exit 1
fi
grep -F -- ".caix-install-staging" "$HF_ARGV_LOG" >/dev/null || {
  echo "error: hf argv did not use staging" >&2
  exit 1
}

echo "catalog install lock handling ok"
