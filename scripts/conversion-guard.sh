#!/usr/bin/env bash
# Guard long-running Core AI conversion queues from overlapping heavy jobs.
# Usage:
#   scripts/conversion-guard.sh          # print active jobs; exit 1 if busy
#   scripts/conversion-guard.sh --wait   # block until no active conversion/generation job remains
set -euo pipefail

wait_mode=0
interval=30
json=0

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
    -h|--help)
      cat <<'EOF'
Guard long-running Core AI conversion queues from overlapping heavy jobs.
Usage:
  scripts/conversion-guard.sh          # print active jobs; exit 1 if busy
  scripts/conversion-guard.sh --wait   # block until no active conversion/generation job remains
  scripts/conversion-guard.sh --json   # emit machine-readable status
EOF
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

active_jobs() {
  ps -axo pid=,ppid=,etime=,pcpu=,pmem=,command= | awk '
    /awk / { next }
    /conversion-guard\.sh/ { next }
    /python\/converter\/convert\.py/ && !/--convert-script/ { print; next }
    /coreai\.llm\.export/ { print; next }
    /(coreai-pipeline|\/caix|\.build\/release\/caix) run / { print; next }
  '
}

print_jobs() {
  jobs="$1"
  if [ "$json" -eq 1 ]; then
    python3 -c '
import json, sys
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
print(json.dumps({"busy": bool(rows), "jobs": rows}, indent=2))
' <<< "$jobs"
  elif [ -n "$jobs" ]; then
    printf '%s\n' "$jobs"
  fi
}

while true; do
  jobs="$(active_jobs || true)"
  if [ -z "$jobs" ]; then
    if [ "$json" -eq 1 ]; then
      printf '{"busy":false,"jobs":[]}\n'
    else
      echo "idle"
    fi
    exit 0
  fi

  if [ "$wait_mode" -eq 0 ]; then
    print_jobs "$jobs"
    exit 1
  fi

  if [ "$json" -eq 0 ]; then
    echo "busy; waiting ${interval}s"
    print_jobs "$jobs"
  fi
  sleep "$interval"
done
