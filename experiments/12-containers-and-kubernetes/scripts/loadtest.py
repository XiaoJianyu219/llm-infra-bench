"""Day12 压测器。纯标准库，能在 WSL / 任何有 python3 的地方跑。

产出的是**带时间戳的时序**，不是一个平均值——3.5 要的是扩容时间线，
必须能把 QPS / p99 的变化与 kubectl 的事件对齐到秒。

用法：
  python3 loadtest.py --url http://localhost:30080/predict --img testdata/LD_000016.png \
      --concurrency 8 --duration 300 --out results/loadtest.jsonl
"""
import argparse, json, os, threading, time, uuid
import urllib.request

STOP = threading.Event()
LOCK = threading.Lock()
BUCKET = []          # 本秒内完成的 (latency, ok)


def post_multipart(url, field, filename, data, timeout):
    b = uuid.uuid4().hex
    body = (b"--" + b.encode() + b"\r\n"
            b'Content-Disposition: form-data; name="' + field.encode()
            + b'"; filename="' + filename.encode() + b'"\r\n'
            b"Content-Type: application/octet-stream\r\n\r\n" + data
            + b"\r\n--" + b.encode() + b"--\r\n")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "multipart/form-data; boundary=" + b)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def worker(url, payload, name, timeout):
    while not STOP.is_set():
        t0 = time.perf_counter()
        try:
            st, _ = post_multipart(url, "file", name, payload, timeout)
            ok = (st == 200)
        except Exception:
            ok = False
        dt = time.perf_counter() - t0
        with LOCK:
            BUCKET.append((dt, ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--img", required=True)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--duration", type=int, default=300)
    ap.add_argument("--timeout", type=float, default=120)
    ap.add_argument("--out", default="results/loadtest.jsonl")
    a = ap.parse_args()

    payload = open(a.img, "rb").read()
    name = os.path.basename(a.img)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    ths = [threading.Thread(target=worker, args=(a.url, payload, name, a.timeout),
                            daemon=True) for _ in range(a.concurrency)]
    t_start = time.time()
    for t in ths:
        t.start()
    print("[%s] 开始加压 concurrency=%d" % (time.strftime("%H:%M:%S"), a.concurrency),
          flush=True)
    with open(a.out, "w") as f:
        while time.time() - t_start < a.duration:
            time.sleep(1.0)
            with LOCK:
                cur, BUCKET[:] = list(BUCKET), []
            lat = sorted(x for x, _ in cur)
            ok = sum(1 for _, o in cur if o)
            rec = dict(t=round(time.time() - t_start, 1),
                       wall=time.strftime("%H:%M:%S"),
                       qps=len(cur), ok=ok, err=len(cur) - ok,
                       p50=round(lat[len(lat) // 2], 3) if lat else None,
                       p99=round(lat[max(0, int(len(lat) * 0.99) - 1)], 3) if lat else None,
                       concurrency=a.concurrency)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print("t=%6.1fs  qps=%3d  ok=%3d err=%3d  p50=%s p99=%s"
                  % (rec["t"], rec["qps"], rec["ok"], rec["err"], rec["p50"], rec["p99"]),
                  flush=True)
    STOP.set()
    print("[%s] 停止加压" % time.strftime("%H:%M:%S"), flush=True)


if __name__ == "__main__":
    main()
