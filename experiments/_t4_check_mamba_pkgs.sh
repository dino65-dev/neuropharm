#!/bin/bash
PY=/home/zeus/miniconda3/envs/cloudspace/bin/python
$PY -m pip list 2>/dev/null | grep -iE "mamba|causal|conv|triton" | head -10
echo "---"
# Try triton (often pre-installed)
$PY -c "import triton; print('triton', triton.__version__)" 2>&1 | head -3
echo "---"
# Is there a prebuilt mamba wheel we can use?
ls /home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages/ 2>/dev/null | grep -iE "mamba|causal" | head -5
