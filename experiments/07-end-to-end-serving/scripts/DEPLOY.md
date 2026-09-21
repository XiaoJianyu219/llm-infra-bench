# 启动与复现

## 已有 AutoDL 环境

在仓库目录执行（多步操作已封装为脚本）：

```bash
bash experiments/07-end-to-end-serving/scripts/serve.sh
```

另一个终端用 `curl --data-binary @image.png -H 'Content-Type: image/png' http://127.0.0.1:18007/predict`。响应 `detections` 每行是 `[x1,y1,x2,y2,score,class_id]`，坐标为原图像素，类别 `LD,FJ,CL,DD`。输入为原始图片请求体。`/healthz` 与 `/metrics` 可直接 GET。`Ctrl-C` 退出会排空已接收队列、停止唯一推理线程并释放 context；不要用后台裸 `&` 留进程。

服务配置 `MAX_BATCH=1/4/8/16`、`MAX_WAIT_MS=0/2/5/10`、`ENGINE` 路径。仅一个 worker；不能用 uvicorn 多 worker 共享显存预算。队列上限256，过载返回503。metrics 累计请求计数不清零，分位数保留最近100000样本；带 `_batch_ms` 的段是该请求所在整批的耗时，不是均摊单图耗时。服务为本机实验接口，不含鉴权、TLS或互联网部署验证。

## Docker：配方交付，尚未构建运行验证

AutoDL 当前没有 docker 命令，不能声称镜像通过运行验证。
基础镜像 `nvcr.io/nvidia/tensorrt:26.08-py3` 对应 TensorRT 11.2.1.2；本次实验是11.3.0.99，两者不同。[NVIDIA 镜像说明](https://docs.nvidia.com/deeplearning/frameworks/container-release-notes/index.html)。

TensorRT plan 与 GPU、TRT 版本和驱动环境相关，**换机器或版本必须重建并复核输出**。不可直接把本实验 engine 拿进上述镜像。模型和 engine 不进入镜像；挂载 ONNX，在目标 GPU 与目标镜像内构建。构建配方中的 Python 包获取与兼容性也尚未在 Docker 验证。

```bash
docker build -t day07-detector experiments/07-end-to-end-serving/scripts
docker run --rm --gpus all -v /absolute/model-dir:/models day07-detector python3 /app/container_build.py /models/day07_batch_fp16.onnx /models/day07_batch_fp16.engine
MODEL_DIR=/absolute/model-dir docker compose -f experiments/07-end-to-end-serving/scripts/docker-compose.yml up
```

模型目录需预先放入本任务导出的 `day07_batch_fp16.onnx`。不要挂载原始私有模型来绕过环境验证。Docker 结果不可与本报告实测直接混用。

## 完整测量

- `setup.sh`：核验、任务目录依赖安装、引擎重建。已存在结果时先归档 Day7，避免覆盖历史实验。
- `measure.sh`：同进程配对七段计时、全800图 CPU/GPU/plugin对齐、动态batch核验。
- `sweep.sh`：客户端容量基准、64配置×3次HTTP压测、逐点GPU采样。耗时较长，在tmux运行；结束自动停服务。
- `build.py` 的插件失败按任务书允许记录；若 `day07_plugin.engine` 不存在只执行GPU NMS路线。
- `simplify` 返回值不是4、标签寻址异常或客户端容量不足，停止核查，不能把异常结果写成成功。
