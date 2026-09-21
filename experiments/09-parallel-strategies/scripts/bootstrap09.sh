#!/bin/bash
# 新双卡实例的一次性环境搭建。分阶段，每阶段可单独重跑。
# 用法: bash bootstrap09.sh probe | repo | venv | deps | model | check
set -u
STAGE="${1:-probe}"
WORK=/root/autodl-tmp/work
REPO=$WORK/llm-infra-bench
VENV=/root/autodl-tmp/venvs/det
TGZ=/root/autodl-tmp/repo_backup.tgz

case "$STAGE" in

probe)
  echo "=== GPU ==="
  nvidia-smi -L
  nvidia-smi --query-gpu=index,name,memory.total,pci.bus_id --format=csv
  echo "=== 拓扑（看两卡之间是 PIX/PHB/NODE/SYS） ==="
  nvidia-smi topo -m 2>&1 | head -12
  echo "=== 驱动 / CUDA ==="
  nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1
  nvcc --version 2>/dev/null | tail -2
  echo "=== 系统 python / conda 里的 torch ==="
  for P in /root/miniconda3/bin/python /usr/bin/python3 $VENV/bin/python; do
    [ -x "$P" ] && echo -n "$P -> " && $P -c "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.device_count())" 2>&1 | tail -1
  done
  echo "=== 磁盘 ==="
  df -h /root/autodl-tmp | tail -1
  echo "=== 已有目录 ==="
  ls /root/autodl-tmp/ 2>/dev/null
  echo "=== 工具 ==="
  which tmux git nvidia-smi; echo "tar: $(which tar)"
  ;;

repo)
  # 从 $TGZ 还原整个仓库（含未推送的 Day 08）。不走 GitHub，避免凭据问题。
  mkdir -p $WORK
  [ -f "$TGZ" ] || { echo "缺少 $TGZ，先从老实例拉过来"; exit 1; }
  tar xzf "$TGZ" -C $WORK
  echo "--- 还原结果 ---"
  ls $REPO
  ls $REPO/days
  echo "day08 results: $(find $REPO/days/day08/results -type f 2>/dev/null | wc -l) 个文件"
  echo "day09 目录: $(ls $REPO/days/day09 2>/dev/null)"
  cd $REPO && git log --oneline -3 2>/dev/null
  ;;

venv)
  # 优先复用备份里带过来的 venv；不行就基于系统 torch 建一个带 system-site-packages 的
  if [ -x "$VENV/bin/python" ] && $VENV/bin/python -c "import torch" 2>/dev/null; then
    echo "复用已有 venv: $($VENV/bin/python -c 'import torch;print(torch.__version__)')"
  else
    BASE=""
    for P in /root/miniconda3/bin/python /usr/bin/python3; do
      if [ -x "$P" ] && $P -c "import torch" 2>/dev/null; then BASE=$P; break; fi
    done
    mkdir -p /root/autodl-tmp/venvs
    if [ -n "$BASE" ]; then
      echo "基于 $BASE 建带 system-site-packages 的 venv（复用它的 torch，省下几个 GB 下载）"
      $BASE -m venv --system-site-packages $VENV
    else
      echo "系统里没有可用 torch，建干净 venv 并安装 torch（会比较慢）"
      python3 -m venv $VENV
      $VENV/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cu124
    fi
  fi
  $VENV/bin/python -c "import torch;print('torch',torch.__version__,torch.version.cuda,'gpus',torch.cuda.device_count())"
  ;;

deps)
  export DS_BUILD_OPS=0
  $VENV/bin/python -m pip freeze > /root/autodl-tmp/pipfreeze_before.txt 2>/dev/null
  $VENV/bin/python -m pip install --quiet --disable-pip-version-check \
      transformers accelerate deepspeed peft modelscope 2>&1 | tail -20
  echo "--- 版本 ---"
  $VENV/bin/python - <<'PY'
import torch, transformers, accelerate
print("torch", torch.__version__, torch.version.cuda, "gpus", torch.cuda.device_count())
print("transformers", transformers.__version__, "accelerate", accelerate.__version__)
for m in ("deepspeed", "peft"):
    try:
        mod = __import__(m); print(m, mod.__version__)
    except Exception as e:
        print(m, "FAILED:", type(e).__name__, e)
PY
  ;;

model)
  M=/root/autodl-tmp/models/Qwen3-0.6B
  if [ -f "$M/config.json" ]; then echo "已存在: $(du -sh $M)"; exit 0; fi
  unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true
  mkdir -p /root/autodl-tmp/models
  nohup $VENV/bin/python -c "
from modelscope import snapshot_download
print(snapshot_download('Qwen/Qwen3-0.6B', local_dir='$M'))
" > /root/autodl-tmp/models/dl_qwen06.log 2>&1 &
  echo "后台下载中，pid $! ，日志 /root/autodl-tmp/models/dl_qwen06.log"
  ;;

check)
  echo "=== 双卡可见性 ==="
  $VENV/bin/python -c "import torch;print('device_count',torch.cuda.device_count());[print(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
  echo "=== P2P ==="
  $VENV/bin/python -c "
import torch
n=torch.cuda.device_count()
for i in range(n):
    for j in range(n):
        if i!=j: print(i,'->',j,'p2p',torch.cuda.can_device_access_peer(i,j))
"
  echo "=== torchrun 在不在 ==="
  ls -l $VENV/bin/torchrun 2>&1
  echo "=== 模型 ==="
  ls /root/autodl-tmp/models/Qwen3-0.6B/config.json 2>&1
  echo "=== day09 脚本 ==="
  ls $REPO/days/day09/scripts 2>&1
  echo "=== 语法自检 ==="
  for f in $REPO/days/day09/scripts/*.py; do $VENV/bin/python -m py_compile "$f" || echo "FAIL $f"; done
  echo compile ok
  bash -n $REPO/days/day09/scripts/run_day09.sh && echo "run_day09.sh syntax ok"
  ;;

*)
  echo "用法: bash bootstrap09.sh probe|repo|venv|deps|model|check"; exit 2;;
esac
