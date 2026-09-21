"""Day12 载体：不依赖 GPU 的检测服务（ONNX Runtime CPU），接口与 Day 07 保持一致。

  POST /predict   接图片，返回检测框
  GET  /healthz   存活。**极轻量**，不碰模型、不加锁——livenessProbe 打这里
  GET  /readyz    就绪。模型加载完成才 200——readinessProbe 打这里
  GET  /metrics   Prometheus 文本格式

/healthz 与 /readyz 必须语义不同（任务书 3.3）：
  * 模型在后台线程里加载，加载期间 /healthz 立刻 200（进程活着），/readyz 503（还不能接流量）。
    这样 livenessProbe 不会在慢加载期间把容器杀掉，readinessProbe 又能挡住流量。
  * 若把 liveness 也打到 /readyz 或 /predict，慢加载就会变成 CrashLoopBackOff——
    这正是 3.3 要复现的第一个失败场景。

依赖刻意压到最小（镜像体积是本日的产出之一）：
  fastapi + uvicorn + numpy + onnxruntime + Pillow。
  **不用 opencv-python**：只为解码图片就要带进整套 OpenCV 与 libGL/libglib，约 90 MB 起。
  **不用 prometheus_client**：指标格式是纯文本，手写几十行即可，省一个依赖。

环境变量（k8s 里由 ConfigMap 注入）：
  MODEL_PATH        onnx 路径，默认 /models/best.onnx
  CONF_THRES        置信度阈值，默认 0.25
  IOU_THRES         NMS IoU 阈值，默认 0.45
  MAX_DET           每图最多返回框数，默认 300
  STARTUP_DELAY_SEC 人为延长模型加载，用于复现 3.3 的失败场景，默认 0
  ORT_THREADS       onnxruntime 线程数，默认 0（交给 ORT 自己定）
"""
import io, os, threading, time
from collections import defaultdict

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/best.onnx")
CONF_THRES = float(os.environ.get("CONF_THRES", "0.25"))
IOU_THRES = float(os.environ.get("IOU_THRES", "0.45"))
MAX_DET = int(os.environ.get("MAX_DET", "300"))
STARTUP_DELAY_SEC = float(os.environ.get("STARTUP_DELAY_SEC", "0"))
ORT_THREADS = int(os.environ.get("ORT_THREADS", "0"))

app = FastAPI(title="sod-onnx-cpu", version="1.0")

# ------------------------------------------------------------------ 指标
_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_lock = threading.Lock()
_counters = defaultdict(float)
_hists = {k: [0] * (len(_BUCKETS) + 1) for k in ("preprocess", "infer", "postprocess", "total")}
_hist_sum = defaultdict(float)
_inflight = 0


def observe(stage, sec):
    with _lock:
        h = _hists[stage]
        for i, b in enumerate(_BUCKETS):
            if sec <= b:
                h[i] += 1
                break
        else:
            h[-1] += 1
        _hist_sum[stage] += sec


def incr(name, v=1.0):
    with _lock:
        _counters[name] += v


# ------------------------------------------------------------------ 模型
class Model:
    def __init__(self):
        self.sess = None
        self.err = None
        self.input_name = None
        self.hw = (1024, 1024)
        self.load_started = time.time()
        self.load_done = None

    def load(self):
        try:
            if STARTUP_DELAY_SEC > 0:
                # 3.3 场景一：人为把加载拉长，用来验证不配 startupProbe 的后果
                time.sleep(STARTUP_DELAY_SEC)
            so = ort.SessionOptions()
            if ORT_THREADS > 0:
                so.intra_op_num_threads = ORT_THREADS
                so.inter_op_num_threads = 1
            s = ort.InferenceSession(MODEL_PATH, sess_options=so,
                                     providers=["CPUExecutionProvider"])
            inp = s.get_inputs()[0]
            self.input_name = inp.name
            shape = inp.shape
            if isinstance(shape[2], int) and isinstance(shape[3], int):
                self.hw = (shape[2], shape[3])
            self.sess = s
        except Exception as e:                       # 加载失败也要让 /healthz 活着报错
            self.err = "%s: %s" % (type(e).__name__, e)
        finally:
            self.load_done = time.time()


M = Model()
threading.Thread(target=M.load, daemon=True).start()


# ------------------------------------------------------------------ 前后处理
def letterbox(im, hw):
    """等比缩放 + 居中填充，返回 (chw_float, scale, pad_x, pad_y)。"""
    H, W = hw
    w0, h0 = im.size
    r = min(W / w0, H / h0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (W, H), (114, 114, 114))
    px, py = (W - nw) // 2, (H - nh) // 2
    canvas.paste(im, (px, py))
    a = np.asarray(canvas, dtype=np.float32) / 255.0      # HWC
    return np.ascontiguousarray(a.transpose(2, 0, 1))[None], r, px, py


def nms(boxes, scores, iou_thres):
    """纯 numpy NMS。boxes 为 xyxy。"""
    idx = scores.argsort()[::-1]
    keep = []
    while idx.size:
        i = idx[0]
        keep.append(int(i))
        if idx.size == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[idx[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[idx[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[idx[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[idx[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        a_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        a_o = ((boxes[idx[1:], 2] - boxes[idx[1:], 0])
               * (boxes[idx[1:], 3] - boxes[idx[1:], 1]))
        iou = inter / (a_i + a_o - inter + 1e-9)
        idx = idx[1:][iou <= iou_thres]
    return keep


def decode(out, r, px, py, w0, h0):
    """输出形如 [1, 4+nc, N]（YOLOv8 风格：前 4 行是 cxcywh，其余是各类分数）。"""
    p = out[0]                                   # [4+nc, N]
    nc = p.shape[0] - 4
    cx, cy, w, h = p[0], p[1], p[2], p[3]
    cls = p[4:4 + nc]
    conf = cls.max(axis=0)
    cid = cls.argmax(axis=0)
    m = conf >= CONF_THRES
    if not m.any():
        return []
    cx, cy, w, h, conf, cid = cx[m], cy[m], w[m], h[m], conf[m], cid[m]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    boxes[:, [0, 2]] -= px                        # 去掉 letterbox 的填充
    boxes[:, [1, 3]] -= py
    boxes /= r                                    # 还原到原图尺度
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w0)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h0)
    keep = nms(boxes, conf, IOU_THRES)[:MAX_DET]
    return [dict(box=[round(float(v), 2) for v in boxes[i]],
                 score=round(float(conf[i]), 4), cls=int(cid[i])) for i in keep]


# ------------------------------------------------------------------ 端点
@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    """存活探针：只证明进程还能响应 HTTP。不碰模型、不加锁、不做任何 IO。

    livenessProbe 打这里。若打到 /predict，高负载下探针超时会把健康的 Pod 杀掉
    （任务书 3.3 的第二个失败场景）。"""
    return "ok"


@app.get("/readyz")
def readyz():
    if M.sess is not None:
        return JSONResponse({"ready": True,
                             "load_sec": round(M.load_done - M.load_started, 3),
                             "model": MODEL_PATH, "input": list(M.hw)})
    code = 503
    body = {"ready": False, "reason": M.err or "loading",
            "elapsed_sec": round(time.time() - M.load_started, 3)}
    return JSONResponse(body, status_code=code)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global _inflight
    if M.sess is None:
        incr("requests_total{status=\"503\"}")
        return JSONResponse({"error": "model not ready"}, status_code=503)
    with _lock:
        _inflight += 1
    t0 = time.perf_counter()
    try:
        raw = await file.read()
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        w0, h0 = im.size
        t1 = time.perf_counter()
        x, r, px, py = letterbox(im, M.hw)
        t2 = time.perf_counter()
        out = M.sess.run(None, {M.input_name: x})[0]
        t3 = time.perf_counter()
        dets = decode(out, r, px, py, w0, h0)
        t4 = time.perf_counter()
        observe("preprocess", t2 - t1)
        observe("infer", t3 - t2)
        observe("postprocess", t4 - t3)
        observe("total", t4 - t0)
        incr("requests_total{status=\"200\"}")
        return {"detections": dets, "n": len(dets),
                "timing_ms": {"decode": round((t1 - t0) * 1e3, 2),
                              "preprocess": round((t2 - t1) * 1e3, 2),
                              "infer": round((t3 - t2) * 1e3, 2),
                              "postprocess": round((t4 - t3) * 1e3, 2)}}
    except Exception as e:
        incr("requests_total{status=\"500\"}")
        return JSONResponse({"error": "%s: %s" % (type(e).__name__, e)}, status_code=500)
    finally:
        with _lock:
            _inflight -= 1


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lines = []
    lines.append("# HELP sod_requests_total 处理过的请求数")
    lines.append("# TYPE sod_requests_total counter")
    with _lock:
        counters = dict(_counters)
        inflight = _inflight
        hists = {k: list(v) for k, v in _hists.items()}
        sums = dict(_hist_sum)
    for k, v in sorted(counters.items()):
        lines.append("sod_%s %d" % (k.replace("requests_total", "requests_total", 1), int(v)))
    lines.append("# HELP sod_inflight_requests 进行中的请求数（HPA 真正该用的指标之一）")
    lines.append("# TYPE sod_inflight_requests gauge")
    lines.append("sod_inflight_requests %d" % inflight)
    lines.append("# HELP sod_ready 模型是否加载完成")
    lines.append("# TYPE sod_ready gauge")
    lines.append("sod_ready %d" % (1 if M.sess is not None else 0))
    for stage in ("preprocess", "infer", "postprocess", "total"):
        h = hists[stage]
        lines.append("# HELP sod_%s_seconds 各段耗时" % stage)
        lines.append("# TYPE sod_%s_seconds histogram" % stage)
        cum = 0
        for i, b in enumerate(_BUCKETS):
            cum += h[i]
            lines.append('sod_%s_seconds_bucket{le="%s"} %d' % (stage, b, cum))
        cum += h[-1]
        lines.append('sod_%s_seconds_bucket{le="+Inf"} %d' % (stage, cum))
        lines.append("sod_%s_seconds_sum %.6f" % (stage, sums.get(stage, 0.0)))
        lines.append("sod_%s_seconds_count %d" % (stage, cum))
    return "\n".join(lines) + "\n"
