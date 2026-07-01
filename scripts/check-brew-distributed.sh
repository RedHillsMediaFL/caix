#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-brew-distributed.sh [--caix <path>] [--ready] [--manifest <path>] [--endpoint <target>...] [--allow-warnings]

Checks the Homebrew-installed caix surface needed before Thunderbolt distributed tests.
It does not start workers or run inference. If endpoints are supplied, it also runs
caix deploy verify with machine identity, version, and link-speed blocker warnings.
USAGE
}

caix_binary="${caix_bin:-caix}"
require_ready=0
manifest=""
endpoints=()
min_machines=2
speed_bytes=4194304
min_mbps=500
max_latency_ms=20
allow_warnings=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --caix) caix_binary="${2:?}"; shift 2 ;;
    --ready) require_ready=1; shift ;;
    --manifest) manifest="${2:?}"; shift 2 ;;
    --endpoint|-e) endpoints+=("${2:?}"); shift 2 ;;
    --endpoints) IFS=',' read -r -a more_endpoints <<< "${2:?}"; endpoints+=("${more_endpoints[@]}"); shift 2 ;;
    --min-machines) min_machines="${2:?}"; shift 2 ;;
    --speed-bytes) speed_bytes="${2:?}"; shift 2 ;;
    --min-mbps) min_mbps="${2:?}"; shift 2 ;;
    --max-latency-ms) max_latency_ms="${2:?}"; shift 2 ;;
    --allow-warnings) allow_warnings=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) echo "error: unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "$caix_binary" == */* ]]; then
  [[ -x "$caix_binary" ]] || {
    echo "error: caix binary not found or not executable: $caix_binary" >&2
    exit 1
  }
else
  caix_binary="$(command -v "$caix_binary")" || {
    echo "error: caix binary not found: $caix_binary" >&2
    exit 1
  }
fi

check_local_warning_fail_on_warn() {
  local tmpdir server_pid port caix_version rc
  tmpdir="$(mktemp -d "${TMPDIR:-/tmp}/caix-deploy-verify-warning.XXXXXX")"
  server_pid=""
  cleanup_local_warning() {
    if [[ -n "$server_pid" ]]; then
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
      server_pid=""
    fi
    rm -rf "$tmpdir"
  }
  trap cleanup_local_warning EXIT INT TERM

  caix_version="$("$caix_binary" --version | awk 'NR == 1 {print $2}')"
  cat >"$tmpdir/server.py" <<'PY'
import json
import os
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

version = os.environ["CAIX_TEST_VERSION"]

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/api/server":
            body = {
                "ok": True,
                "name": "coreai-pipeline",
                "caix_version": version,
                "machine_name": socket.gethostname(),
                "runtime_linked": True,
                "compute_unit": "gpu",
            }
            data = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_response(404)
        self.end_headers()

server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
with open(os.environ["CAIX_TEST_PORT_FILE"], "w", encoding="utf-8") as fh:
    fh.write(str(server.server_address[1]))
    fh.write("\n")
server.serve_forever()
PY
  CAIX_TEST_VERSION="$caix_version" CAIX_TEST_PORT_FILE="$tmpdir/port" \
    python3 "$tmpdir/server.py" >"$tmpdir/server.out" 2>"$tmpdir/server.err" &
  server_pid="$!"

  for _ in {1..100}; do
    [[ -s "$tmpdir/port" ]] && break
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "error: deploy verify warning test server exited" >&2
      sed -n '1,80p' "$tmpdir/server.err" >&2 || true
      exit 1
    fi
    sleep 0.05
  done
  [[ -s "$tmpdir/port" ]] || {
    echo "error: deploy verify warning test server did not start" >&2
    exit 1
  }
  port="$(<"$tmpdir/port")"

  "$caix_binary" deploy verify --endpoint "http://127.0.0.1:$port" \
    --min-machines 1 --no-speed-test --max-latency-ms 10000 \
    --json >"$tmpdir/diagnostic.json"
  CAIX_VERIFY_JSON="$tmpdir/diagnostic.json" python3 - <<'PY'
import json
import os
import sys

doc = json.load(open(os.environ["CAIX_VERIFY_JSON"], encoding="utf-8"))
if doc.get("ok") is not True:
    sys.exit("diagnostic local warning probe should be ok without --fail-on-warn")
warnings = "\n".join(doc.get("warnings", []))
if "local machine endpoint" not in warnings:
    sys.exit("diagnostic local warning probe did not produce local-machine warning")
PY

  set +e
  "$caix_binary" deploy verify --endpoint "http://127.0.0.1:$port" \
    --min-machines 1 --no-speed-test --max-latency-ms 10000 \
    --fail-on-warn --json >"$tmpdir/fail-on-warn.json"
  rc="$?"
  set -e
  [[ "$rc" -ne 0 ]] || {
    echo "error: deploy verify --fail-on-warn ignored local-machine warning" >&2
    exit 1
  }
  CAIX_VERIFY_JSON="$tmpdir/fail-on-warn.json" python3 - <<'PY'
import json
import os
import sys

doc = json.load(open(os.environ["CAIX_VERIFY_JSON"], encoding="utf-8"))
if doc.get("ok") is not False:
    sys.exit("--fail-on-warn local warning probe should not be ok")
warnings = "\n".join(doc.get("warnings", []))
if "local machine endpoint" not in warnings:
    sys.exit("--fail-on-warn local warning probe did not report local-machine warning")
PY
  cleanup_local_warning
  trap - EXIT INT TERM
}

"$caix_binary" --version
"$caix_binary" doctor --no-fail
"$caix_binary" cluster plan --help >/dev/null
"$caix_binary" deploy verify --help >/dev/null

if [[ -n "$manifest" ]]; then
  json="$("$caix_binary" cluster plan --manifest "$manifest" --workers main=4,mini=2 --json)"
  CLUSTER_PLAN_JSON="$json" python3 - <<'PY'
import json
import os
import sys

doc = json.loads(os.environ["CLUSTER_PLAN_JSON"])
runtime = doc.get("runtime_plan")
if doc.get("dry_run") is not True or not isinstance(runtime, dict):
    sys.exit(1)
roles = [stage.get("role") for stage in runtime.get("stages", [])]
if roles != ["embeddings", "transformer_layers", "transformer_layers", "final_norm_head"]:
    sys.exit(1)
if not isinstance(runtime.get("total_layer_count"), int) or runtime["total_layer_count"] <= 0:
    sys.exit(1)
if doc.get("position_mode") != "full_prefix":
    sys.exit(1)
if runtime.get("position_mode") != doc["position_mode"]:
    sys.exit(1)
boundary = doc.get("boundary_tensor")
if not isinstance(boundary, dict):
    sys.exit(1)
if boundary.get("name") != "hidden_states":
    sys.exit(1)
shape = boundary.get("shape")
if not (
    isinstance(shape, list)
    and len(shape) == 3
    and shape[0] == 1
    and shape[1] == -1
    and isinstance(shape[2], int)
    and shape[2] > 0
):
    sys.exit(1)
if boundary.get("scalar_type") != "float16":
    sys.exit(1)
if runtime.get("boundary_tensor") != boundary:
    sys.exit(1)
PY
fi

if [[ "$require_ready" == "1" ]]; then
  "$caix_binary" cluster join --help >/dev/null
  "$caix_binary" --help | grep -q -- '--cluster'
  "$caix_binary" --help | grep -q -- '--kv-capacity <N>'
  "$caix_binary" --help | grep -q -- '--headroom-gb <GB>'
  "$caix_binary" serve --help 2>/dev/null | grep -q -- '--prompt-tokens'
  "$caix_binary" serve --help 2>/dev/null | grep -q -- '--join-timeout'
  "$caix_binary" cluster join --help 2>/dev/null | grep -q -- '--connect-timeout'
  "$caix_binary" deploy verify --help 2>/dev/null | grep -q -- '--speed-bytes'
  "$caix_binary" deploy verify --help 2>/dev/null | grep -q -- '--min-mbps'
  "$caix_binary" deploy verify --help 2>/dev/null | grep -q -- '--fail-on-warn'
  check_local_warning_fail_on_warn
fi

if [[ "${#endpoints[@]}" -gt 0 ]]; then
  args=(deploy verify --min-machines "$min_machines" --speed-bytes "$speed_bytes" \
    --min-mbps "$min_mbps" --max-latency-ms "$max_latency_ms")
  if [[ "$allow_warnings" != "1" ]]; then
    args+=(--fail-on-warn)
  fi
  for endpoint in "${endpoints[@]}"; do
    [[ -n "$endpoint" ]] && args+=(--endpoint "$endpoint")
  done
  "$caix_binary" "${args[@]}"
fi

echo "brew distributed surface ok"
