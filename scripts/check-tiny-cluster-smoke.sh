#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/check-tiny-cluster-smoke.sh --manifest <stage-manifest.json> (--print-commands|--local-loopback) [options]

Prepares or runs the minimal staged cluster smoke with an installed caix binary.
Use check-brew-distributed.sh with endpoints first when testing two Macs.

Options:
  --caix <path>              caix binary (default: caix)
  --manifest <path>          staged manifest
  --coordinator <host:port>  coordinator address (default: 127.0.0.1:0)
  --bind-host <host>         coordinator bind host (default: coordinator host)
  --remote-stage <id>        remote stage id; repeatable (default: transformer stages)
  --prompt-tokens <list>     comma-separated token ids (default: 1,2,3)
  --max-tokens <N>           generated token count (default: 1)
  --join-timeout <s>         coordinator join timeout (default: 120)
  --connect-timeout <s>      worker connect timeout (default: 120)
  --worker-root <dir>        copied staged bundle root for printed worker commands
  --worker-manifest <path>   manifest path on worker machines (default: <worker-root>/<manifest-name>)
  --startup-timeout <s>      local coordinator startup timeout (default: 30)
  --min-free-mb <N>          local free-space floor before model load (default: 2048)
  --work-dir <dir>           local-loopback log directory (default: temp dir)
  --lock <path>              local-loopback lock file (default: ${CAIX_HEAVY_TASK_LOCK:-$PWD/.agent-heavy-task.lock})
  --no-lock                  skip local-loopback lock creation
  --print-commands           print coordinator and worker commands only
  --local-loopback           start coordinator and all remote-stage workers locally
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

caix_binary="${caix_bin:-caix}"
manifest=""
coordinator="127.0.0.1:0"
bind_host=""
prompt_tokens="1,2,3"
max_tokens=1
join_timeout=120
connect_timeout=120
worker_root=""
worker_manifest=""
startup_timeout=30
min_free_mb=2048
work_dir=""
lock_path="${CAIX_HEAVY_TASK_LOCK:-$PWD/.agent-heavy-task.lock}"
use_lock=1
print_commands=0
local_loopback=0
remote_stage_filters=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --caix) caix_binary="${2:?}"; shift 2 ;;
    --manifest) manifest="${2:?}"; shift 2 ;;
    --coordinator) coordinator="${2:?}"; shift 2 ;;
    --bind-host) bind_host="${2:?}"; shift 2 ;;
    --remote-stage) remote_stage_filters+=("${2:?}"); shift 2 ;;
    --prompt-tokens) prompt_tokens="${2:?}"; shift 2 ;;
    --max-tokens) max_tokens="${2:?}"; shift 2 ;;
    --join-timeout) join_timeout="${2:?}"; shift 2 ;;
    --connect-timeout) connect_timeout="${2:?}"; shift 2 ;;
    --worker-root) worker_root="${2:?}"; shift 2 ;;
    --worker-manifest) worker_manifest="${2:?}"; shift 2 ;;
    --startup-timeout) startup_timeout="${2:?}"; shift 2 ;;
    --min-free-mb) min_free_mb="${2:?}"; shift 2 ;;
    --work-dir) work_dir="${2:?}"; shift 2 ;;
    --lock) lock_path="${2:?}"; shift 2 ;;
    --no-lock) use_lock=0; shift ;;
    --print-commands) print_commands=1; shift ;;
    --local-loopback) local_loopback=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "error: unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) echo "error: unexpected argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$manifest" ]] || { usage >&2; exit 2; }
[[ "$print_commands" == "1" || "$local_loopback" == "1" ]] || {
  echo "error: choose --print-commands or --local-loopback" >&2
  usage >&2
  exit 2
}
[[ "$print_commands" == "0" || "$local_loopback" == "0" ]] || {
  echo "error: choose only one of --print-commands or --local-loopback" >&2
  usage >&2
  exit 2
}
if [[ -n "$worker_root" && "$print_commands" != "1" ]]; then
  die "--worker-root is only valid with --print-commands"
fi
if [[ -n "$worker_manifest" && "$print_commands" != "1" ]]; then
  die "--worker-manifest is only valid with --print-commands"
fi

case "$coordinator" in
  *:*) ;;
  *) die "--coordinator must be host:port" ;;
esac
coordinator_host="${coordinator%:*}"
coordinator_port="${coordinator##*:}"
[[ -n "$coordinator_host" ]] || die "--coordinator host is empty"
[[ "$coordinator_port" =~ ^[0-9]+$ ]] || die "--coordinator port must be numeric"
(( coordinator_port >= 0 && coordinator_port <= 65535 )) || die "--coordinator port is out of range"
if [[ -z "$bind_host" ]]; then
  bind_host="$coordinator_host"
fi
if [[ "$print_commands" == "1" && "$coordinator_port" == "0" ]]; then
  die "--print-commands needs a concrete --coordinator host:port"
fi

case "$max_tokens" in ''|*[!0-9]*) die "--max-tokens must be a positive integer" ;; esac
(( max_tokens > 0 )) || die "--max-tokens must be a positive integer"
case "$min_free_mb" in ''|*[!0-9]*) die "--min-free-mb must be a positive integer" ;; esac
(( min_free_mb > 0 )) || die "--min-free-mb must be a positive integer"

if [[ "$caix_binary" == */* ]]; then
  [[ -x "$caix_binary" ]] || die "caix binary not found or not executable: $caix_binary"
else
  caix_binary="$(command -v "$caix_binary")" || die "caix binary not found: $caix_binary"
fi

manifest="$(python3 - "$manifest" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"
[[ -f "$manifest" ]] || die "manifest not found: $manifest"

stage_ids=()
stage_paths=()
decode_stage_paths=()
parser_args=("$manifest")
if [[ "${#remote_stage_filters[@]}" -gt 0 ]]; then
  parser_args+=("${remote_stage_filters[@]}")
fi
manifest_rows="$(python3 - "${parser_args[@]}" <<'PY'
from pathlib import Path
import json
import sys

manifest = Path(sys.argv[1]).expanduser().resolve()
filters = set(sys.argv[2:])
with manifest.open("r", encoding="utf-8") as handle:
    doc = json.load(handle)
stages = doc.get("stages")
if stages is None:
    stages = doc.get("cluster", {}).get("stages", [])
if not isinstance(stages, list):
    raise SystemExit("manifest stages must be an array")

base = manifest.parent
seen = set()
rows = []
for stage in stages:
    if not isinstance(stage, dict):
        continue
    stage_id = stage.get("id")
    role = stage.get("role")
    bundle = stage.get("bundle")
    if role != "transformer_layers":
        continue
    if filters and stage_id not in filters:
        continue
    if not isinstance(stage_id, str) or not stage_id:
        raise SystemExit("transformer stage is missing id")
    if not isinstance(bundle, str) or not bundle:
        raise SystemExit(f"stage {stage_id} is missing bundle")
    seen.add(stage_id)
    bundle_path = Path(bundle).expanduser()
    if not bundle_path.is_absolute():
        bundle_path = base / bundle_path
    decode_asset = stage.get("decode_asset") or ""
    decode_path = ""
    if decode_asset:
        decode_path_obj = Path(decode_asset).expanduser()
        if not decode_path_obj.is_absolute():
            decode_path_obj = base / decode_path_obj
        decode_path = str(decode_path_obj.resolve())
    rows.append((stage_id, str(bundle_path.resolve()), decode_path))

missing = sorted(filters - seen)
if missing:
    raise SystemExit("unknown remote stage: " + ", ".join(missing))
if not rows:
    raise SystemExit("manifest has no selected transformer stages")
for row in rows:
    print("\t".join(row))
PY
)" || die "failed to parse remote stages from manifest"

while IFS=$'\t' read -r stage_id stage_path decode_stage_path; do
  [[ -n "$stage_id" ]] || continue
  stage_ids+=("$stage_id")
  stage_paths+=("$stage_path")
  decode_stage_paths+=("$decode_stage_path")
done <<< "$manifest_rows"
[[ "${#stage_ids[@]}" -gt 0 ]] || die "manifest has no selected transformer stages"

for stage_path in "${stage_paths[@]}"; do
  [[ -d "$stage_path" ]] || die "stage bundle not found: $stage_path"
done

print_cmd() {
  printf '  '
  while [[ $# -gt 0 ]]; do
    printf '%q' "$1"
    shift
    [[ $# -eq 0 ]] || printf ' '
  done
  printf '\n'
}

worker_path() {
  local source_path="$1"
  if [[ -z "$worker_root" ]]; then
    printf '%s\n' "$source_path"
    return
  fi
  python3 - "$manifest" "$worker_root" "$source_path" <<'PY'
from pathlib import Path, PurePosixPath
import sys

manifest = Path(sys.argv[1]).resolve(strict=False)
root = sys.argv[2].rstrip("/")
source = Path(sys.argv[3]).resolve(strict=False)
try:
    relative = source.relative_to(manifest.parent)
except ValueError:
    relative = Path(source.name)
print(str(PurePosixPath(root) / PurePosixPath(*relative.parts)))
PY
}

worker_manifest_path="$manifest"
if [[ -n "$worker_manifest" ]]; then
  worker_manifest_path="$worker_manifest"
elif [[ -n "$worker_root" ]]; then
  worker_manifest_path="$(worker_path "$manifest")"
fi

serve_command=(
  "$caix_binary" serve
  --cluster "$manifest"
  --host "$bind_host"
  --port "$coordinator_port"
  --prompt-tokens "$prompt_tokens"
  --max-tokens "$max_tokens"
  --join-timeout "$join_timeout"
  --once
)
for stage_id in "${stage_ids[@]}"; do
  serve_command+=(--remote-stage "$stage_id")
done

if [[ "$print_commands" == "1" ]]; then
  echo "# verify links first with check-brew-distributed.sh endpoint speed checks"
  echo "# coordinator"
  print_cmd "${serve_command[@]}"
  echo "# workers; run one per selected transformer stage, split across machines as needed"
  if [[ -n "$worker_root" ]]; then
    echo "# worker commands assume the staged bundle was copied to: $worker_root"
  fi
  for i in "${!stage_ids[@]}"; do
    worker_stage_path="$(worker_path "${stage_paths[$i]}")"
    join_command=(
      "$caix_binary" cluster join
      --coordinator "$coordinator"
      --manifest "$worker_manifest_path"
      --stage "$worker_stage_path"
      --stage-id "${stage_ids[$i]}"
      --connect-timeout "$connect_timeout"
    )
    if [[ -n "${decode_stage_paths[$i]}" ]]; then
      worker_decode_stage_path="$(worker_path "${decode_stage_paths[$i]}")"
      join_command+=(--decode-stage "$worker_decode_stage_path")
    fi
    print_cmd "${join_command[@]}"
  done
  exit 0
fi

check_free_space() {
  local target="$1"
  local available_kib
  available_kib="$(df -Pk "$target" | awk 'NR == 2 {print $4}')"
  [[ "$available_kib" =~ ^[0-9]+$ ]] || die "could not read free space for $target"
  local required_kib=$((min_free_mb * 1024))
  if (( available_kib < required_kib )); then
    die "free space below ${min_free_mb}MB at $target"
  fi
}

lock_acquired=0
coord_pid=""
worker_pids=()
cleanup() {
  if [[ -n "$coord_pid" ]] && kill -0 "$coord_pid" >/dev/null 2>&1; then
    kill "$coord_pid" >/dev/null 2>&1 || true
  fi
  for pid in "${worker_pids[@]}"; do
    if kill -0 "$pid" >/dev/null 2>&1; then
      kill "$pid" >/dev/null 2>&1 || true
    fi
  done
  if [[ "$lock_acquired" == "1" ]]; then
    rm -f "$lock_path"
  fi
}
trap cleanup EXIT INT TERM

check_free_space "$(dirname "$manifest")"
if [[ "$use_lock" == "1" ]]; then
  mkdir -p "$(dirname "$lock_path")"
  if ( set -C; printf 'pid=%s\nscript=check-tiny-cluster-smoke\n' "$$" >"$lock_path" ) 2>/dev/null; then
    lock_acquired=1
  else
    die "heavy-task lock exists: $lock_path"
  fi
fi

if [[ -z "$work_dir" ]]; then
  work_dir="$(mktemp -d "${TMPDIR:-/tmp}/caix-cluster-smoke.XXXXXX")"
else
  mkdir -p "$work_dir"
fi

coord_stdout="$work_dir/coordinator.out"
coord_stderr="$work_dir/coordinator.err"
"${serve_command[@]}" >"$coord_stdout" 2>"$coord_stderr" &
coord_pid="$!"

actual_port="$coordinator_port"
deadline=$((SECONDS + startup_timeout))
while [[ "$actual_port" == "0" ]]; do
  if ! kill -0 "$coord_pid" >/dev/null 2>&1; then
    die "coordinator exited before listening; logs: $work_dir"
  fi
  if line="$(awk '/cluster coordinator listening on / { print; exit }' "$coord_stderr" 2>/dev/null)" && [[ -n "$line" ]]; then
    actual_port="$(printf '%s\n' "$line" | sed -n 's/.*:\([0-9][0-9]*\);.*/\1/p')"
    [[ -n "$actual_port" ]] || die "could not parse coordinator port; logs: $work_dir"
    break
  fi
  (( SECONDS < deadline )) || die "timed out waiting for coordinator; logs: $work_dir"
  sleep 0.2
done

actual_coordinator="${coordinator_host}:${actual_port}"
for i in "${!stage_ids[@]}"; do
  worker_stdout="$work_dir/worker-${stage_ids[$i]}.out"
  worker_stderr="$work_dir/worker-${stage_ids[$i]}.err"
  join_command=(
    "$caix_binary" cluster join
    --coordinator "$actual_coordinator"
    --manifest "$manifest"
    --stage "${stage_paths[$i]}"
    --stage-id "${stage_ids[$i]}"
    --connect-timeout "$connect_timeout"
  )
  if [[ -n "${decode_stage_paths[$i]}" ]]; then
    join_command+=(--decode-stage "${decode_stage_paths[$i]}")
  fi
  "${join_command[@]}" >"$worker_stdout" 2>"$worker_stderr" &
  worker_pids+=("$!")
done

if ! wait "$coord_pid"; then
  coord_pid=""
  die "coordinator failed; logs: $work_dir"
fi
coord_pid=""

for pid in "${worker_pids[@]}"; do
  wait "$pid" >/dev/null 2>&1 || true
done

expected_remote="$(IFS=,; echo "${stage_ids[*]}")"
python3 - "$coord_stdout" "$expected_remote" "$max_tokens" <<'PY'
import json
import sys

path, expected_csv, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(path, "r", encoding="utf-8") as handle:
    doc = json.load(handle)
expected = sorted(filter(None, expected_csv.split(",")))
remote = sorted(doc.get("remote_stage_ids", []))
if remote != expected:
    raise SystemExit(f"remote stages mismatch: expected {expected}, got {remote}")
if doc.get("generated_token_count") != max_tokens:
    raise SystemExit(
        f"generated_token_count mismatch: expected {max_tokens}, got {doc.get('generated_token_count')}"
    )
if doc.get("stop_reason") != "max_tokens":
    raise SystemExit(f"unexpected stop_reason: {doc.get('stop_reason')}")
print("tiny cluster smoke ok: remote_stages=" + ",".join(remote) + f" generated_token_count={max_tokens}")
PY

echo "logs: $work_dir"
