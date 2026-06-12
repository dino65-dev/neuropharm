#!/bin/bash
set -e
PY=/home/zeus/miniconda3/envs/cloudspace/bin/python
# Install mamba-ssm and causal-conv1d
$PY -m pip install --quiet causal-conv1d mamba-ssm 2>&1 | tail -5
# Test
$PY -c "
try:
    import mamba_ssm
    print('mamba_ssm OK', mamba_ssm.__version__ if hasattr(mamba_ssm, '__version__') else 'imported')
except Exception as e:
    print('mamba_ssm error:', e)
try:
    from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn
    print('rmsnorm_fn OK')
except Exception as e:
    print('rmsnorm_fn error:', e)
" 2>&1 | head -10
