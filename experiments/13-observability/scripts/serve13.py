"""Day13 3.1 的被观测对象：在 Day07 的推理服务上加一层 Prometheus 指标。

复用 Day07 的 pipeline（decode/preprocess/nms/restore/TRTModel），不重写推理逻辑——
本日不做新实验，服务本身的行为要和 Day07 一致，只是把内部状态按 Prometheus 文本格式暴露出来。

暴露的指标（对应任务书 3.1 的 panel 清单）：
  sod_requests_total{status}            counter    -> QPS = rate(), 错误率
  sod_request_duration_seconds          histogram  -> p50/p95/p99 = histogram_quantile()
  sod_stage_duration_seconds{stage}     histogram  -> 七段里各段的分位数
  sod_inflight_requests                 gauge      -> 在途请求数（推理服务真正的压力指标）
  sod_queue_depth                       gauge      -> 队列深度
  sod_batch_size                        histogram  -> 动态批处理的批大小分布
  sod_gpu_utilization_percent           gauge      -> 由后台 nvidia-smi 采样
  sod_gpu_memory_used_bytes             gauge      -> 同上，口径见 report

histogram 的 bucket 边界按 Day07 实测的延迟范围定（p50 从 26 ms 到 1460 ms 都出现过），
不覆盖实际范围的话 histogram_quantile 会给出严重失真的分位数——这是任务书第 4 节第 2 条的坑。
"""
import asyncio, os, time, collections, threading, subprocess, math
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import PlainTextResponse

# ---------------------------------------------------------------- 指标原语
# 手写 Prometheus 文本格式：venv 里没有 prometheus_client，而且手写能完全控制 bucket 边界。

# 端到端延迟的 bucket（秒）。依据：Day07 §6 实测 p50 26.21 ms(conc1) -> 1460.30 ms(conc64)，
# p99 34.81 ms -> 1492.50 ms。两端各留一档余量。
E2E_BUCKETS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2,
               0.3, 0.5, 0.75, 1.0, 2.0, 5.0]
# 单段延迟更小：Day07 §3 最大单段 preprocess 15.441 ms，最小 restore 0.105 ms。
STAGE_BUCKETS = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25]
# 批大小：MAX_BATCH 最大 16
BATCH_BUCKETS = [1, 2, 3, 4, 6, 8, 12, 16]


class Hist:
    def __init__(self, buckets):
        self.buckets = list(buckets)
        self.counts = [0] * (len(self.buckets) + 1)   # 最后一个是 +Inf
        self.sum = 0.0
        self.n = 0
        self.lock = threading.Lock()

    def observe(self, v):
        with self.lock:
            self.sum += v
            self.n += 1
            for i, b in enumerate(self.buckets):
                if v <= b:
                    self.counts[i] += 1
                    break
            else:
                self.counts[-1] += 1

    def render(self, name, labels=""):
        # Prometheus 的 histogram 是累积桶，这里把逐桶计数累加起来
        lines, acc = [], 0
        inner = labels[1:-1] if labels else ""
        for i, b in enumerate(self.buckets):
            acc += self.counts[i]
            sep = "," if inner else ""
            lines.append('%s_bucket{%s%sle="%s"} %d' % (name, inner, sep, _fmt(b), acc))
        acc += self.counts[-1]
        sep = "," if inner else ""
        lines.append('%s_bucket{%s%sle="+Inf"} %d' % (name, inner, sep, acc))
        lines.append('%s_sum%s %.6f' % (name, labels, self.sum))
        lines.append('%s_count%s %d' % (name, labels, self.n))
        return lines


def _fmt(v):
    if isinstance(v, int) or float(v).is_integer():
        return str(int(v))
    return repr(float(v))


class M:
    requests = collections.Counter()          # status -> n
    e2e = Hist(E2E_BUCKETS)
    stages = collections.defaultdict(lambda: Hist(STAGE_BUCKETS))
    batch = Hist(BATCH_BUCKETS)
    inflight = 0
    lock = threading.Lock()
    gpu_util = 0.0
    gpu_mem = 0.0
    gpu_samples = 0


def gpu_sampler(period=1.0):
    """后台采 nvidia-smi。注意口径：这是整卡的 used memory，含 CUDA context，
    与 Day08 区分的 max_memory_allocated / reserved 不是同一档。"""
    q = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
         "--format=csv,noheader,nounits", "-i", "0"]
    while True:
        try:
            out = subprocess.run(q, capture_output=True, text=True, timeout=5).stdout.strip()
            u, m = out.split(",")
            M.gpu_util = float(u.strip())
            M.gpu_mem = float(m.strip()) * 1024 * 1024
            M.gpu_samples += 1
        except Exception:
            pass
        time.sleep(period)


# ---------------------------------------------------------------- 服务
class Service:
    def __init__(self):
        self.max_batch = int(os.getenv("MAX_BATCH", "1"))
        self.wait = float(os.getenv("MAX_WAIT_MS", "0")) / 1000
        self.queue = asyncio.Queue(maxsize=int(os.getenv("QUEUE_MAX", "256")))
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference-owner")
        self.engine = None
        self.ready = False
        self.task = None

    def setup(self):
        import torch, cv2
        from pipeline import TRTModel
        torch.set_num_threads(int(os.getenv("TORCH_THREADS", "4")))
        cv2.setNumThreads(1)
        self.engine = TRTModel(os.environ["ENGINE"])
        for _ in range(24):
            self.engine(torch.zeros(self.max_batch, 3, 1024, 1024, device="cuda"))

    def run_batch(self, items):
        import torch, numpy as np
        from pipeline import decode, preprocess, nms, restore
        arrays, metas, timings = [], [], []
        for blob, _, arrival in items:
            t = time.perf_counter(); im = decode(blob)
            d = time.perf_counter(); a, meta = preprocess(im)
            p = time.perf_counter()
            arrays.append(a); metas.append(meta)
            timings.append(dict(queue=t - arrival, decode=d - t, preprocess=p - d))
        t = time.perf_counter()
        x = torch.from_numpy(np.stack(arrays)).cuda(); torch.cuda.synchronize()
        h = time.perf_counter()
        y = self.engine(x); torch.cuda.synchronize()
        f = time.perf_counter()
        dets = [nms(z) for z in y]; torch.cuda.synchronize()
        n = time.perf_counter()
        dets = [z.cpu() for z in dets]; torch.cuda.synchronize()
        d2 = time.perf_counter()
        nb = len(items)
        out = []
        for det, meta, tm in zip(dets, metas, timings):
            a0 = time.perf_counter(); v = restore(det, meta).tolist(); b0 = time.perf_counter()
            # 批内分摊：批级耗时按批大小均摊到每个请求，否则 batch>1 时单请求的段耗时会被高估
            tm.update(h2d=(h - t) / nb, inference=(f - h) / nb,
                      nms=(n - f) / nb, d2h=(d2 - n) / nb, restore=b0 - a0)
            out.append((dict(detections=v), tm))
        return out

    async def worker(self):
        loop = asyncio.get_running_loop()
        while True:
            first = await self.queue.get()
            items = [first]
            if self.max_batch > 1:
                deadline = time.perf_counter() + self.wait
                while len(items) < self.max_batch:
                    remain = deadline - time.perf_counter()
                    if self.wait <= 0:
                        if self.queue.empty():
                            break
                        items.append(self.queue.get_nowait())
                    else:
                        if remain <= 0:
                            break
                        try:
                            items.append(await asyncio.wait_for(self.queue.get(), remain))
                        except asyncio.TimeoutError:
                            break
            M.batch.observe(len(items))
            try:
                results = await loop.run_in_executor(self.pool, self.run_batch, items)
                for (blob, fut, arrival), (body, tm) in zip(items, results):
                    if not fut.done():
                        fut.set_result((body, tm))
            except Exception as e:                       # noqa: BLE001
                for _, fut, _ in items:
                    if not fut.done():
                        fut.set_exception(e)


service = Service()


@asynccontextmanager
async def lifespan(app):
    threading.Thread(target=gpu_sampler, daemon=True).start()
    await asyncio.get_running_loop().run_in_executor(None, service.setup)
    service.ready = True
    service.task = asyncio.create_task(service.worker())
    yield
    service.task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return dict(ready=service.ready, max_batch=service.max_batch,
                max_wait_ms=service.wait * 1000, gpu_samples=M.gpu_samples)


@app.post("/predict")
async def predict(request: Request):
    if not service.ready:
        M.requests["503"] += 1
        raise HTTPException(503, "Not ready")
    body = await request.body()
    if not body or len(body) > 16 * 1024 * 1024:
        M.requests["413"] += 1
        raise HTTPException(413, "bad body size")
    t0 = time.perf_counter()
    with M.lock:
        M.inflight += 1
    fut = asyncio.get_running_loop().create_future()
    try:
        try:
            service.queue.put_nowait((body, fut, t0))
        except asyncio.QueueFull:
            M.requests["429"] += 1
            raise HTTPException(429, "queue full")
        out, tm = await fut
        for k, v in tm.items():
            M.stages[k].observe(v)
        M.e2e.observe(time.perf_counter() - t0)
        M.requests["200"] += 1
        return out
    except HTTPException:
        raise
    except Exception as e:                                # noqa: BLE001
        M.requests["500"] += 1
        raise HTTPException(500, "%s: %s" % (type(e).__name__, e))
    finally:
        with M.lock:
            M.inflight -= 1


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    L = []
    L.append("# HELP sod_requests_total 按状态码分的请求计数")
    L.append("# TYPE sod_requests_total counter")
    for st, n in sorted(M.requests.items()):
        L.append('sod_requests_total{status="%s"} %d' % (st, n))
    if not M.requests:
        L.append('sod_requests_total{status="200"} 0')

    L.append("# HELP sod_request_duration_seconds 端到端延迟（进 handler 到出 handler）")
    L.append("# TYPE sod_request_duration_seconds histogram")
    L += M.e2e.render("sod_request_duration_seconds")

    L.append("# HELP sod_stage_duration_seconds 七段中各段耗时，批级段已按批大小均摊")
    L.append("# TYPE sod_stage_duration_seconds histogram")
    for st in sorted(M.stages):
        L += M.stages[st].render("sod_stage_duration_seconds", '{stage="%s"}' % st)

    L.append("# HELP sod_batch_size 动态批处理实际成批的大小分布")
    L.append("# TYPE sod_batch_size histogram")
    L += M.batch.render("sod_batch_size")

    L.append("# HELP sod_inflight_requests 在途请求数（已进 handler 未返回）")
    L.append("# TYPE sod_inflight_requests gauge")
    L.append("sod_inflight_requests %d" % M.inflight)

    L.append("# HELP sod_queue_depth 批处理队列里等待的请求数")
    L.append("# TYPE sod_queue_depth gauge")
    L.append("sod_queue_depth %d" % service.queue.qsize())

    L.append("# HELP sod_gpu_utilization_percent nvidia-smi 的 utilization.gpu")
    L.append("# TYPE sod_gpu_utilization_percent gauge")
    L.append("sod_gpu_utilization_percent %.1f" % M.gpu_util)

    L.append("# HELP sod_gpu_memory_used_bytes nvidia-smi 口径的整卡已用显存（含 CUDA context）")
    L.append("# TYPE sod_gpu_memory_used_bytes gauge")
    L.append("sod_gpu_memory_used_bytes %.0f" % M.gpu_mem)

    L.append("# HELP sod_service_ready 模型是否加载完成")
    L.append("# TYPE sod_service_ready gauge")
    L.append("sod_service_ready %d" % (1 if service.ready else 0))
    return "\n".join(L) + "\n"
