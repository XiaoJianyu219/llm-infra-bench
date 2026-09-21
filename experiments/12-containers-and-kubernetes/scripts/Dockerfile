# 对比行「多阶段 + 瘦身」：本日实际部署用的就是这一个。
#
# 这一版是**按实测改过的**。第一版我把"所有想得到的瘦身技巧"都堆上去，
# 结果 452 MB，反而比单纯多阶段的 434 MB 还大 18 MB。拆开看是两处自伤：
#   1. apt 装了 curl —— k8s 的 httpGet 探针是 kubelet 在容器外发的，
#      容器里根本不需要 curl。装它纯属条件反射。
#   2. RUN chown -R svc:svc /app —— chown 会让整个 /app 在新层里重存一份。
#      改成 COPY --chown 在拷贝时就设好属主，不额外增层。
#
# 多阶段本身省了多少、清缓存省了多少，见 results/image_variants.txt 的四格对照。

ARG PIP_INDEX_URL=https://pypi.org/simple/

# ---------- builder ----------
FROM python:3.12-slim AS builder
ARG PIP_INDEX_URL
WORKDIR /w
COPY requirements.txt .
# --no-cache-dir：不落 pip 缓存。多阶段下 builder 的缓存本来也不会进最终镜像，
# 但加上它能让 builder 层本身更小、构建更快。
# --prefix=/install：装到独立前缀，便于整目录拷进运行镜像
RUN pip install --no-cache-dir --prefix=/install --index-url "$PIP_INDEX_URL" --trusted-host mirrors.aliyun.com -r requirements.txt

# ---------- runtime ----------
FROM python:3.12-slim AS runtime
# 不再 apt install 任何东西：运行期只需要 python 与依赖
COPY --from=builder /install /usr/local
WORKDIR /app
# 建用户放在 COPY 之前，且用 COPY --chown 设属主，避免多出一个 chown 层
RUN useradd -m -u 10001 svc
COPY --chown=svc:svc app.py .
USER svc
ENV MODEL_PATH=/models/best.onnx PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
