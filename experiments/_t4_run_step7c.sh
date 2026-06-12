#!/bin/bash
set -e
export HF_TOKEN=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
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
PY=/home/zeus/miniconda3/envs/cloudspace/bin/python
$PY step7c_deeper_layers.py
