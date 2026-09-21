# Recipe only: this AutoDL container has no Docker executable/daemon.
# TRT 26.08 contains 11.2.1.2, unlike the experiment's 11.3.0.99.
# Build a NEW engine inside this image on the deployment GPU from mounted ONNX.
FROM nvcr.io/nvidia/tensorrt:26.08-py3
WORKDIR /app
RUN python3 -m pip install --no-cache-dir torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
RUN python3 -m pip install --no-cache-dir fastapi==0.141.1 uvicorn==0.53.0 opencv-python-headless==4.12.0.88 numpy
COPY common.py pipeline.py trt_utils.py server.py container_build.py /app/
ENV ENGINE=/models/day07_batch_fp16.engine MAX_BATCH=4 MAX_WAIT_MS=2 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
