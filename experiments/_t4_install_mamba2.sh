#!/bin/bash
PY=/home/zeus/miniconda3/envs/cloudspace/bin/python
$PY -m pip install mamba-ssm 2>&1 | tail -30
