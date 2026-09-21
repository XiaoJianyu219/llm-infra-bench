#!/bin/bash
# 2.4-C3：逐 rank 保存 RNG 后再做一次弹性重启，看两个 rank 是否都能逐位对齐
set -u
cd /root/autodl-tmp/work/llm-infra-bench/days/day11
TR=/root/autodl-tmp/venvs/det/bin/torchrun
export CUBLAS_WORKSPACE_CONFIG=:4096:8
ED=/root/autodl-tmp/train/elastic_v2
rm -rf $ED; mkdir -p $ED
echo "########## 2.4-C3 弹性重启 + 逐 rank RNG + 严格确定 ##########"; date "+%F %T"
( timeout 400 $TR --nproc_per_node=2 --max-restarts=2 --master_port=29831   scripts/train_ft.py --strategy ddp --micro-bsz 2 --grad-accum 2 --steps 10   --deterministic 1 --per-rank-rng 1 --save-every 2 --ckpt-level c2   --save-path $ED/ck.pt --auto-resume $ED --kill-at 5 --kill-marker $ED/killed.marker   --tag killC3 --out results/runs/kill_C3.json ) > results/logs/kill_C3.log 2>&1
grep -E "auto-resume|从 .* 恢复|rank1 pid=|step [0-9]+ done|exitcode" results/logs/kill_C3.log | head -30
echo "--- 掉卡前后 step 4 两个 rank 的 loss 是否逐位一致 ---"
/root/autodl-tmp/venvs/det/bin/python - <<PY
import re
L=open("results/logs/kill_C3.log").read().splitlines()
seen={}
for ln in L:
    for m in re.finditer(r"\[rank(\d)\] step (\d+) done loss ([0-9.]+)", ln):
        r,s,v=m.group(1),int(m.group(2)),m.group(3)
        seen.setdefault((r,s),[]).append(v)
for (r,s),vs in sorted(seen.items()):
    if len(vs)>1:
        print("rank%s step %d 出现 %d 次：%s -> %s" % (r,s,len(vs),vs,
              "逐位一致" if len(set(vs))==1 else "不一致"))
PY
sleep 4; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "C3DONE $(date '+%F %T')"
