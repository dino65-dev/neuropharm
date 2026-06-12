#!/bin/bash
set -euo pipefail
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
if [ -z "${HF_TOKEN:-}" ]; then
  if [ -f ~/.hf_token ]; then
    export HF_TOKEN="$(cat ~/.hf_token)"
  else
    echo "HF_TOKEN is not set and ~/.hf_token does not exist" >&2
    exit 2
  fi
fi
export HF_TOKEN
PY=/home/zeus/miniconda3/envs/cloudspace/bin/python
$PY -m experiments.step8_qwen35_layer_sweep
