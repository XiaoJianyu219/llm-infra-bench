#!/bin/bash
# Day13 收尾：补上面板压测的噪声地板（report 第 6 节标注的未完项）。
# 同一配置重复 3 次（第 1 次是 00:xx 那轮，本脚本再跑 2 轮），用极差作为噪声地板。
# 沿用 Day10 的纪律：每轮都是独立进程，不是同一进程内的多个 block。
set -u
B=/root/autodl-tmp/day13
D=/root/autodl-tmp/work/llm-infra-bench/days/day07
cd $B
step(){ echo; echo "########## $* ##########"; date "+%H:%M:%S"; }

step "1. 重新起服务与 Prometheus（实例重启后进程已丢）"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$D/scripts:$B:/root/autodl-tmp/sod/day07-deps
export YOLO_CONFIG_DIR=/root/autodl-tmp/sod/day06-yolo-config
export MPLCONFIGDIR=/root/autodl-tmp/sod/day07-mpl
export XDG_CACHE_HOME=/root/autodl-tmp/sod/day07-cache
export TMPDIR=/root/autodl-tmp/sod/day07-tmp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$TMPDIR" results
export ENGINE=/root/autodl-tmp/sod/day07_batch_fp16.engine
export MAX_BATCH=8 MAX_WAIT_MS=2
cd $D/scripts
setsid nohup /root/autodl-tmp/venvs/det/bin/python -m uvicorn serve13:app \
  --host 127.0.0.1 --port 18013 --no-access-log > $B/results/serve13_r2.log 2>&1 < /dev/null &
cd $B
for i in $(seq 1 60); do
  curl -s --max-time 3 http://127.0.0.1:18013/healthz | grep -q '"ready":true' && { echo "服务就绪"; break; }
  sleep 3
done
setsid nohup $B/prometheus/prometheus --config.file=$B/prometheus.yml \
  --storage.tsdb.path=$B/tsdb --storage.tsdb.retention.time=6h \
  --web.listen-address=127.0.0.1:19090 > $B/results/prometheus_r2.log 2>&1 < /dev/null &
sleep 8
curl -s --max-time 5 http://127.0.0.1:19090/api/v1/targets | grep -o '"health":"[a-z]*"' | head -1

IMG=$(find /root/autodl-tmp/sod/dataset800 -name '*.png' | head -1)
echo "测试图: $IMG"

for R in 2 3; do
  step "2.$R 第 $R 轮（与第 1 轮完全同配置）"
  /root/autodl-tmp/venvs/det/bin/python loadgen13.py \
    --url http://127.0.0.1:18013/predict --img "$IMG" \
    --stages 1:60,2:60,4:60,8:60,16:60 --idle-tail 30 \
    --out results/loadgen_r$R.jsonl 2>&1 | grep -E 'T\+|LOADGEN_DONE'
  /root/autodl-tmp/venvs/det/bin/python analyze13.py results/loadgen_r$R.jsonl 2>&1 | tail -22
  mv results/panel_numbers.json results/panel_numbers_r$R.json
  mv results/panel_series.json results/panel_series_r$R.json
done
echo NOISE13_DONE $(date "+%F %T")
