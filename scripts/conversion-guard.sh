#!/usr/bin/env bash
# Guard long-running Core AI conversion queues from overlapping heavy jobs.
# Usage:
#   scripts/conversion-guard.sh          # print active jobs; exit 1 if busy
#   scripts/conversion-guard.sh --wait   # block until no heavy job or heavy-task lock remains
#   scripts/conversion-guard.sh --ignore-lock
#                                      # ignore a stale lock, but still reject active jobs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/lib/caix-env.sh"

LOCK="$(caix_env caix_heavy_task_lock HEAVY_TASK_LOCK "$REPO_DIR/.agent-heavy-task.lock")"
wait_mode=0
interval=30
json=0
ignore_lock=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --wait)
      wait_mode=1
      shift
      ;;
    --interval)
      interval="${2:?missing value for --interval}"
      shift 2
      ;;
    --json)
      json=1
      shift
      ;;
    --ignore-lock)
      ignore_lock=1
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Guard long-running Core AI conversion queues from overlapping heavy jobs.
Usage:
  scripts/conversion-guard.sh          # print active jobs; exit 1 if busy
  scripts/conversion-guard.sh --wait   # block until no heavy job or heavy-task lock remains
  scripts/conversion-guard.sh --json   # emit machine-readable status
  scripts/conversion-guard.sh --ignore-lock
                                      # ignore a stale lock, but still reject active jobs
EOF
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

process_table() {
  if [ -n "${caix_test_process_table-}" ]; then
    printf '%s\n' "$caix_test_process_table"
    return 0
  fi
  ps -axo pid=,ppid=,etime=,pcpu=,pmem=,command=
}

active_jobs() {
  if [ -n "${caix_test_active_jobs-}" ]; then
    printf '%s\n' "$caix_test_active_jobs"
    return 0
  fi
  process_table | awk '
    /awk / { next }
    /conversion-guard\.sh/ { next }
    /python\/converter\/convert\.py/ && !/--convert-script/ { print; next }
    /coreai\.llm\.export/ { print; next }
    /hf (download|upload|upload-large-folder)/ { print; next }
    /(^|[[:space:]\/])git-lfs([[:space:]]|$)/ { print; next }
    /(^|[[:space:]\/])git lfs (fetch|pull|push|checkout|prune|migrate|upload|download)/ { print; next }
    /\.build\/(debug|release)\/caix (run|eagle)/ { print; next }
    /(^|\/)(caix|coreai-pipeline) (run|eagle)/ { print; next }
    /(^|\/)swift (build|test)|swift-package|swiftc|swift-frontend|xctest/ { print; next }
  '
}

redact_job_commands() {
  python3 -c '
import re
import sys

options = [
    "--" + "token",
    "--" + "auth-token",
    "--hf-" + "token",
    "--api-key",
    "--access-key",
    "--secret",
    "--password",
]

def redact(command):
    for option in options:
        command = re.sub(
            rf"({re.escape(option)}=)(\S+)",
            r"\1[redacted]",
            command,
            flags=re.IGNORECASE,
        )
        command = re.sub(
            rf"({re.escape(option)})([ \t]+)(\S+)",
            r"\1\2[redacted]",
            command,
            flags=re.IGNORECASE,
        )
    command = re.sub(
        r"(\b[A-Za-z_][A-Za-z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_KEY|ACCESS_KEY|PRIVATE_KEY)\s*=\s*)(\S+)",
        r"\1[redacted]",
        command,
        flags=re.IGNORECASE,
    )
    command = re.sub(
        r"(?i)(Bearer[ \t]+)(\S+)",
        r"\1[redacted]",
        command,
    )
    return command

for line in sys.stdin:
    parts = line.rstrip("\n").split(None, 5)
    if len(parts) == 6:
        print("{} {} {} {} {} {}".format(*parts[:5], redact(parts[5])))
'
}

print_status() {
  lock_busy="$1"
  jobs="$2"
  if [ "$json" -eq 1 ]; then
    LOCK_BUSY="$lock_busy" LOCK_PATH="$LOCK" python3 -c '
import json
import os
import sys
rows = []
for line in sys.stdin:
    parts = line.rstrip("\n").split(None, 5)
    if len(parts) == 6:
        rows.append({
            "pid": int(parts[0]),
            "ppid": int(parts[1]),
            "elapsed": parts[2],
            "cpu_percent": float(parts[3]),
            "mem_percent": float(parts[4]),
            "command": parts[5],
        })
lock_path = os.environ["LOCK_PATH"] if os.environ.get("LOCK_BUSY") == "1" else None
print(json.dumps({"busy": bool(rows) or lock_path is not None, "lock_path": lock_path, "jobs": rows}, indent=2))
' <<< "$jobs"
  else
    if [ "$lock_busy" -eq 1 ]; then
      printf 'heavy-task lock exists: %s\n' "$LOCK"
    fi
    if [ -n "$jobs" ]; then
      printf '%s\n' "$jobs"
    fi
  fi
}

while true; do
  lock_busy=0
  if [ "$ignore_lock" -eq 0 ] && [ -e "$LOCK" ]; then
    lock_busy=1
  fi
  jobs="$(active_jobs | redact_job_commands || true)"
  if [ "$lock_busy" -eq 0 ] && [ -z "$jobs" ]; then
    if [ "$json" -eq 1 ]; then
      printf '{"busy":false,"jobs":[]}\n'
    else
      echo "idle"
    fi
    exit 0
  fi

  if [ "$wait_mode" -eq 0 ]; then
    print_status "$lock_busy" "$jobs"
    exit 1
  fi

  if [ "$json" -eq 0 ]; then
    echo "busy; waiting ${interval}s"
    print_status "$lock_busy" "$jobs"
  fi
  sleep "$interval"
done
