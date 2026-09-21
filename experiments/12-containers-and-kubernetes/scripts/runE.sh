#!/bin/bash
# Day12 第五轮：直接检验预测 3 里那条可证伪的断言 ——
# "CPU 利用率封顶之后，继续加压时 inflight 仍在线性上升而利用率不再变化"。
# 单副本、无 HPA，并发 1 -> 2 -> 4 -> 8 -> 16 阶梯加压，
# 每 5s 同时采 /metrics 的 inflight 与 kubectl top 的 CPU。
set -u
cd /mnt/d/llm-infra-day12
R=results; mkdir -p $R
step(){ echo; echo "########## $* ##########"; date "+%H:%M:%S"; }

step "1. 单副本、latest 镜像、删掉 HPA（排除扩容干扰）"
kubectl delete hpa sod >/dev/null 2>&1
kubectl apply -f manifests/deployment.yaml >/dev/null
kubectl scale deployment/sod --replicas=1 >/dev/null
kubectl rollout status deployment/sod --timeout=300s 2>&1 | tail -1
for i in $(seq 1 30); do
  [ "$(curl -s -o /dev/null -w %{http_code} --max-time 5 http://localhost:30080/readyz)" = 200 ] && break
  sleep 3
done
echo "requests.cpu = $(kubectl get deploy sod -o jsonpath='{.spec.template.spec.containers[0].resources.requests.cpu}')"
echo "limits.cpu   = $(kubectl get deploy sod -o jsonpath='{.spec.template.spec.containers[0].resources.limits.cpu}')"

# 采样器：每 5s 记一行 "时刻 inflight cpu_millicores"
cat > /tmp/sampler.sh <<'SH'
#!/bin/bash
OUT=$1; DUR=$2
END=$((SECONDS+DUR))
echo "time inflight cpu_m util_vs_requests_pct" > "$OUT"
while [ $SECONDS -lt $END ]; do
  INF=$(curl -s --max-time 4 http://localhost:30080/metrics | awk '/^sod_inflight_requests /{print $2}')
  CPU=$(kubectl top pod -l app=sod --no-headers 2>/dev/null | awk '{gsub("m","",$2); s+=$2} END{print s+0}')
  PCT=$(awk -v c="${CPU:-0}" 'BEGIN{printf "%.0f", c/500*100}')
  echo "$(date +%H:%M:%S) ${INF:-NA} ${CPU:-NA} $PCT" >> "$OUT"
  sleep 5
done
SH
chmod +x /tmp/sampler.sh

step "2. 阶梯加压：并发 1/2/4/8/16，每档 60s"
nohup bash /tmp/sampler.sh $R/runE_inflight_vs_cpu.txt 400 > /dev/null 2>&1 &
SPID=$!
sleep 10
for C in 1 2 4 8 16; do
  echo "--- 并发 $C 开始 $(date +%H:%M:%S) ---"
  python3 loadtest.py --url http://localhost:30080/predict --img testdata/LD_000016.png \
    --concurrency $C --duration 60 --timeout 180 --out $R/runE_c$C.jsonl > /dev/null 2>&1
  echo "--- 并发 $C 结束 $(date +%H:%M:%S) ---"
done
wait $SPID 2>/dev/null

step "3. 结果"
cat $R/runE_inflight_vs_cpu.txt
echo RUNE_DONE $(date "+%F %T")
