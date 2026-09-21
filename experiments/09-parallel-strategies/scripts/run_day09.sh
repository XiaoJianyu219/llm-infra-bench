#!/bin/bash
# Day09 全流程。放 tmux 里跑，输出 tee 到 results/run_all.log。
# 不用 set -e：单个配置 OOM / 失败是结论的一部分，不能让整条链停。
set -u
cd /root/autodl-tmp/work/llm-infra-bench/days/day09
PY=${PY:-/root/autodl-tmp/venvs/det/bin/python}
TORCHRUN=${TORCHRUN:-/root/autodl-tmp/venvs/det/bin/torchrun}
mkdir -p results/runs results/logs

# 工作点：seq 512、每卡 micro 2。
# 为什么不取更大：Day 08 实测同卡同模型在 4096 token/step 就 OOM
# （lm_head 输出 [tokens,151936] 主导）。strong scaling 的单卡分母是
# global batch 4 = 2048 token，必须落在墙以内，否则分母根本跑不出来。
SEQ=${SEQ:-512}
MB=${MB:-2}
ACC=${ACC:-1}
STEPS=${STEPS:-10}
WARM=${WARM:-10}
REP=${REP:-3}

step() { echo; echo "############ $* ############"; date '+%F %T'; }

gpus() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; }

# 任务书 4.1：每个配置跑完必须确认两张卡都回到基线，否则下一个配置的显存是错的
cleanup() {
  pkill -f torchrun 2>/dev/null
  pkill -f train_dist.py 2>/dev/null
  pkill -f comm_bench.py 2>/dev/null
  sleep 4
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')
  echo "  清理后各卡显存: ${used}"
  for u in $used; do
    if [ "$u" -gt 600 ]; then
      echo "  !! 有卡未回到基线（${u} MiB），存在孤儿进程，等 10 秒再查"
      sleep 10
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv
      return 1
    fi
  done
  return 0
}

run1() {   # run1 <tag> <nproc> <strategy> <micro_bsz> <scaling> [额外参数...]
  local tag=$1 np=$2 st=$3 mb=$4 sc=$5; shift 5
  if [ -f "results/runs/${tag}.json" ]; then echo "skip (exists) $tag"; return; fi
  echo "--- $tag : nproc=$np strategy=$st micro=$mb scaling=$sc $* ---"
  local t0=$SECONDS
  if [ "$np" = "1" ]; then
    $PY scripts/train_dist.py --strategy "$st" --seq-len $SEQ --micro-bsz "$mb" \
      --grad-accum $ACC --steps $STEPS --warmup $WARM --repeats $REP --scaling "$sc" \
      "$@" --out "results/runs/${tag}.json" > "results/logs/${tag}.log" 2>&1
  else
    $TORCHRUN --nproc_per_node=$np --master_port=29577 scripts/train_dist.py \
      --strategy "$st" --seq-len $SEQ --micro-bsz "$mb" --grad-accum $ACC \
      --steps $STEPS --warmup $WARM --repeats $REP --scaling "$sc" \
      "$@" --out "results/runs/${tag}.json" > "results/logs/${tag}.log" 2>&1
  fi
  echo "  [$((SECONDS-t0))s] $(grep -E '^\[|OOM|Error' "results/logs/${tag}.log" | tail -2)"
  cleanup
}

step "0. 环境与 GPU 基线"
nvidia-smi > results/nvidia_smi_before.txt 2>&1
gpus
$PY -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'gpus',torch.cuda.device_count())" | tee results/env.txt
$PY -c "import deepspeed;print('deepspeed',deepspeed.__version__)" 2>&1 | tee -a results/env.txt
$PY -c "import transformers,accelerate,peft;print('transformers',transformers.__version__,'accelerate',accelerate.__version__,'peft',peft.__version__)" 2>&1 | tee -a results/env.txt
$PY -m pip list 2>/dev/null >> results/env.txt
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "检测到 $NGPU 张卡"

# ================= 1. 通信带宽（任务书第 1 节，最高优先级，必须最先做） ==========
step "1a. NCCL transport（NCCL_DEBUG=INFO，原文进 report）"
if [ "$NGPU" -ge 2 ]; then
  NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV \
    $TORCHRUN --nproc_per_node=2 --master_port=29578 scripts/nccl_probe.py \
    > results/logs/nccl_debug.log 2>&1
  echo "--- 关键行 ---"
  grep -iE "via |transport|P2P|SHM|NET|Ring|Channel|Connected" results/logs/nccl_debug.log \
    | head -30 | tee results/nccl_transport.txt
  cleanup
else
  echo "只有 $NGPU 张卡，跳过（本项需要 2 卡）" | tee results/nccl_transport.txt
fi

step "1b. 集合通信带宽扫描 1MB→1GB"
if [ "$NGPU" -ge 2 ]; then
  $TORCHRUN --nproc_per_node=2 --master_port=29579 scripts/comm_bench.py \
    --out results/comm_bench.json 2>&1 | tee results/comm_bench.log
else
  $PY scripts/comm_bench.py --out results/comm_bench.json 2>&1 | tee results/comm_bench.log
fi
cleanup

# ================= 2. 冷启动逐步耗时曲线（任务书 4.7） =========================
step "2. 冷启动曲线：决定丢几步"
run1 curve_single 1 single $MB na
if [ "$NGPU" -ge 2 ]; then
  run1 curve_ddp 2 ddp $MB weak
fi

# ================= 3. 单卡基准（两个分母） ====================================
step "3. 单卡基准"
run1 "single_mb${MB}"        1 single $MB          weak     # weak scaling 的分母
run1 "single_mb$((MB*2))"    1 single $((MB*2))    strong   # strong scaling 的分母

# ================= 4. 双卡矩阵 ===============================================
step "4. 双卡矩阵（每卡 micro=$MB，global=$((MB*2)) ）"
if [ "$NGPU" -ge 2 ]; then
  for ST in ddp zero1 zero2 zero3 fsdp; do
    run1 "w2_${ST}_mb${MB}" 2 "$ST" $MB weak
  done
else
  echo "只有 $NGPU 张卡，双卡矩阵全部跳过 —— 任务书 3.3 的主体无法完成"
fi

# ================= 4b. LoRA 对照（任务书 3.1 第 4 问 / 3.2 的 LoRA 重算） ======
# 题眼：LoRA 下 ZeRO-1/2 切分的对象缩水到可训练参数那一小块，
# 理论上几乎省不到显存；只有 ZeRO-3 切冻结基座才有意义。这里实测验证。
step "4b. LoRA 对照（基座 bf16 冻结，只训 q/k/v/o 的 adapter）"
LORA_ARGS="--lora --lora-r 16 --lora-alpha 32"
run1 "lora_single_mb${MB}" 1 single $MB weak $LORA_ARGS
if [ "$NGPU" -ge 2 ]; then
  for ST in ddp zero1 zero2 zero3 fsdp; do
    run1 "lora_w2_${ST}_mb${MB}" 2 "$ST" $MB weak $LORA_ARGS
  done
fi

# ================= 5. 汇总 ===================================================
step "5. 汇总与对照表"
$PY scripts/analyze09.py 2>&1 | tee results/summary.log

step "6. 收尾：确认两张卡都归零"
cleanup
nvidia-smi > results/nvidia_smi_after.txt 2>&1
gpus
echo
echo "ALL DONE $(date '+%F %T')"
