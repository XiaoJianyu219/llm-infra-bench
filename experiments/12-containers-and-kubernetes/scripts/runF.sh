#!/bin/bash
# 与 runE 相同的阶梯，但用 sync 镜像（/metrics 不被推理阻塞）+ 内存放到 3Gi（避免 OOM 打断）
set -u
cd /mnt/d/llm-infra-day12
R=results; mkdir -p $R
kubectl delete hpa sod >/dev/null 2>&1
sed -e 's#image: sod-onnx-cpu:latest#image: sod-onnx-cpu:sync#' \
    -e 's#memory: "1Gi"#memory: "3Gi"#' manifests/deployment.yaml > /tmp/dep-f.yaml
kubectl apply -f /tmp/dep-f.yaml >/dev/null
kubectl scale deployment/sod --replicas=1 >/dev/null
kubectl rollout status deployment/sod --timeout=300s 2>&1 | tail -1
for i in $(seq 1 30); do
  [ "$(curl -s -o /dev/null -w %{http_code} --max-time 5 http://localhost:30080/readyz)" = 200 ] && break
  sleep 3
done
nohup bash /tmp/sampler.sh $R/runF_inflight_vs_cpu.txt 400 > /dev/null 2>&1 &
SPID=$!
sleep 10
for C in 1 2 4 8 16; do
  echo "--- 并发 $C  $(date +%H:%M:%S) ---"
  python3 loadtest.py --url http://localhost:30080/predict --img testdata/LD_000016.png \
    --concurrency $C --duration 60 --timeout 180 --out $R/runF_c$C.jsonl > /dev/null 2>&1
done
wait $SPID 2>/dev/null
cat $R/runF_inflight_vs_cpu.txt
echo "--- 各档吞吐 ---"
python3 - <<'PY'
import json,glob,re
for c in (1,2,4,8,16):
    rs=[json.loads(l) for l in open("results/runF_c%d.jsonl"%c) if l.strip()]
    ok=sum(r["ok"] for r in rs); err=sum(r["err"] for r in rs)
    ps=sorted(r["p50"] for r in rs if r["p50"])
    print("并发 %2d  ok=%3d err=%4d  QPS=%.2f  p50中位=%.2fs" %
          (c, ok, err, ok/float(max(len(rs),1)), ps[len(ps)//2] if ps else float('nan')))
PY
echo RUNF_DONE $(date "+%F %T")
