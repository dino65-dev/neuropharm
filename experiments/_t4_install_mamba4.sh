#!/bin/bash
PY=/home/zeus/miniconda3/envs/cloudspace/bin/python
# Try prebuilt mamba_ssm wheel from official repo
# Pre-built wheels are available for specific torch/cuda/python combinations
echo "=== Trying prebuilt mamba_ssm from Dao-AILab repository ==="
$PY -m pip install --no-build-isolation \
    --index-url https://pypi.org/simple/ \
    "mamba-ssm" 2>&1 | tail -5
echo "---"
$PY -c "import mamba_ssm; print('mamba_ssm imported OK')" 2>&1 | head -3
