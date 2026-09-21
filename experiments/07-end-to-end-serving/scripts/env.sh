#!/bin/bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/root/autodl-tmp/sod/day07-deps:/root/autodl-tmp/work/llm-infra-bench/days/day07/scripts
export YOLO_CONFIG_DIR=/root/autodl-tmp/sod/day06-yolo-config
export MPLCONFIGDIR=/root/autodl-tmp/sod/day07-mpl
export XDG_CACHE_HOME=/root/autodl-tmp/sod/day07-cache
export TMPDIR=/root/autodl-tmp/sod/day07-tmp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$TMPDIR"
cd /root/autodl-tmp/sod || exit 1
