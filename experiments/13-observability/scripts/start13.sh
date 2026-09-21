#!/bin/bash
# Day13 3.1：起服务 + Prometheus。AutoDL 容器内起不了 dockerd（Day12 已判定），全部单二进制。
set -u
B=/root/autodl-tmp/day13
D=/root/autodl-tmp/work/llm-infra-bench/days/day07
cd $B
mkdir -p results tsdb
step(){ echo; echo "########## $* ##########"; date "+%H:%M:%S"; }

step "1. 起推理服务（复用 Day07 的 pipeline 与动态 batch 引擎）"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$D/scripts:$B:/root/autodl-tmp/sod/day07-deps
export YOLO_CONFIG_DIR=/root/autodl-tmp/sod/day06-yolo-config
export MPLCONFIGDIR=/root/autodl-tmp/sod/day07-mpl
export XDG_CACHE_HOME=/root/autodl-tmp/sod/day07-cache
export TMPDIR=/root/autodl-tmp/sod/day07-tmp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$TMPDIR"
export ENGINE=/root/autodl-tmp/sod/day07_batch_fp16.engine
export MAX_BATCH=${MAX_BATCH:-8} MAX_WAIT_MS=${MAX_WAIT_MS:-2}
cd $D/scripts
setsid nohup /root/autodl-tmp/venvs/det/bin/python -m uvicorn serve13:app \
  --host 127.0.0.1 --port 18013 --no-access-log > $B/results/serve13.log 2>&1 < /dev/null &
cd $B
for i in $(seq 1 60); do
  R=$(curl -s --max-time 3 http://127.0.0.1:18013/healthz)
  echo "$R" | grep -q '"ready":true' && { echo "服务就绪: $R"; break; }
  sleep 3
done
echo "--- /metrics 前 20 行 ---"
curl -s --max-time 5 http://127.0.0.1:18013/metrics | head -20

step "2. 起 Prometheus（scrape_interval 1s）"
setsid nohup $B/prometheus/prometheus --config.file=$B/prometheus.yml \
  --storage.tsdb.path=$B/tsdb --storage.tsdb.retention.time=6h \
  --web.listen-address=127.0.0.1:19090 --web.enable-admin-api \
  > $B/results/prometheus.log 2>&1 < /dev/null &
sleep 8
echo "--- targets 状态（排障第 2 层）---"
curl -s --max-time 5 'http://127.0.0.1:19090/api/v1/targets' \
  | /root/autodl-tmp/venvs/det/bin/python -c "import sys,json; d=json.load(sys.stdin)['data']['activeTargets']; [print(t['scrapeUrl'], t['health'], t.get('lastError','')) for t in d]" 2>&1 | head -5
echo "--- PromQL 自检（排障第 3 层）---"
curl -s --max-time 5 --get 'http://127.0.0.1:19090/api/v1/query' --data-urlencode 'query=sod_service_ready' \
  | /root/autodl-tmp/venvs/det/bin/python -c "import sys,json; print(json.load(sys.stdin))" 2>&1 | head -3
echo START13_DONE $(date "+%F %T")
