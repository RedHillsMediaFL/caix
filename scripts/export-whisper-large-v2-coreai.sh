#!/bin/bash
set -euo pipefail

caix_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
snapshot="/Volumes/SSD/hf-cache/models--openai--whisper-large-v2/snapshots/ae4642769ce2ad8fc292556ccea8e901f1530655"
coreai_models_repository="/Volumes/SSD/ai-dev/coreai-gemma4/vendor/coreai-models"
authoring_temp_root="/Volumes/SSD/caix/.tmp"
python_bin="/Volumes/SSD/caix/.tmp/coreai-b2-probe-20260722/bin/python"
output="${1:-/Volumes/SSD/caix-kept-exports/speech/whisper-large-v2-fp16.aimodel}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$caix_repo_root/python"

exec "$python_bin" -m whisper_large_v2.convert \
  --snapshot "$snapshot" \
  --source-contract "$caix_repo_root/models/whisper-large-v2-source.json" \
  --authoring-source "$caix_repo_root/models/coreai-models-authoring-source.json" \
  --coreai-models-repository "$coreai_models_repository" \
  --authoring-temp-root "$authoring_temp_root" \
  --output "$output" \
  --export \
  --max-resident-gib 42
