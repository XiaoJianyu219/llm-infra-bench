#!/bin/bash
export PYTHONDONTWRITEBYTECODE=1
export YOLO_CONFIG_DIR=/root/autodl-tmp/sod/day06-yolo-config
export MPLCONFIGDIR=/root/autodl-tmp/sod/day06-mpl
export XDG_CACHE_HOME=/root/autodl-tmp/sod/day06-cache
export TORCH_HOME=/root/autodl-tmp/sod/day06-torch
export TMPDIR=/root/autodl-tmp/sod/day06-tmp
export TORCHINDUCTOR_CACHE_DIR=/root/autodl-tmp/sod/day06-torchinductor
export CUDA_CACHE_PATH=/root/autodl-tmp/sod/day06-cuda-cache
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p "$TMPDIR"
cd /root/autodl-tmp/sod || exit 1
