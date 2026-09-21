#!/bin/bash
# Day12 3.2：把"多阶段"与"清 pip 缓存"拆开，各自量一次。
# 第一轮只测了两端（555MB vs 452MB），中间省下的 103MB 到底归谁，必须分开建才知道。
set -u
cd /mnt/d/llm-infra-day12
R=results; mkdir -p $R
M=https://mirrors.aliyun.com/pypi/simple/
step(){ echo; echo "########## $* ##########"; date "+%F %T"; }

# A：单阶段 + 不清缓存（= sod:slim，已建）
# B：单阶段 + --no-cache-dir       → 只看"清缓存"的贡献
# C：多阶段 + 不清缓存             → 只看"多阶段"的贡献
# D：多阶段 + --no-cache-dir       → 两者都有（= sod-onnx-cpu，已建）

step "B. 单阶段 + --no-cache-dir"
cat > /tmp/Dockerfile.B <<EOF
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --index-url $M --trusted-host mirrors.aliyun.com -r requirements.txt
COPY app.py .
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
docker build -q -f /tmp/Dockerfile.B -t sod:B . && echo built

step "C. 多阶段 + 不清缓存"
cat > /tmp/Dockerfile.C <<EOF
FROM python:3.12-slim AS builder
WORKDIR /w
COPY requirements.txt .
RUN pip install --prefix=/install --index-url $M --trusted-host mirrors.aliyun.com -r requirements.txt
FROM python:3.12-slim AS runtime
COPY --from=builder /install /usr/local
WORKDIR /app
COPY app.py .
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
docker build -q -f /tmp/Dockerfile.C -t sod:C . && echo built

step "四格对照"
{
  printf "%-28s %s\n" "策略" "体积"
  for t in "python:3.12-slim|基座" "sod:slim|A 单阶段+留缓存" "sod:B|B 单阶段+清缓存" \
           "sod:C|C 多阶段+留缓存" "sod-onnx-cpu:latest|D 多阶段+清缓存+非root+apt瘦身"; do
    img=${t%%|*}; desc=${t##*|}
    sz=$(docker images "$img" --format "{{.Size}}" | head -1)
    printf "%-28s %s\n" "$desc" "$sz"
  done
} | tee $R/image_variants.txt

step "各层大小：看 103MB 到底省在哪"
for img in sod:slim sod:B sod:C sod-onnx-cpu:latest; do
  echo "===== $img ====="
  docker history --format "{{.Size}}\t{{.CreatedBy}}" "$img" | head -8 | cut -c1-120
  echo
done | tee $R/docker_history.txt

step "构建上下文（.dockerignore 修好后）"
docker build -f Dockerfile -t sod-ctx-probe . 2>&1 | grep -i "build context" | head -2 | tee $R/build_context.txt

echo VARIANTS_DONE $(date "+%F %T")
