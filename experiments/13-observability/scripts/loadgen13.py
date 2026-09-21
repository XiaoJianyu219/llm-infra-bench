"""Day13 3.1：阶梯加压，让面板上留下"QPS 上升 -> p99 抬头 -> GPU 利用率跟着变"的波形。

不是新实验：参数沿用 Day07 §6/§7 已经扫过的并发档，这里只是为了喂出可判读的时序。
"""
import argparse, json, threading, time, urllib.request, collections, sys


def worker(url, blob, stop, rec, lock):
    req = urllib.request.Request(url, data=blob, method="POST",
                                 headers={"Content-Type": "application/octet-stream"})
    while not stop.is_set():
        t = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                r.read()
            ok, dt = True, time.perf_counter() - t
        except Exception:
            ok, dt = False, time.perf_counter() - t
        with lock:
            rec.append((time.time(), ok, dt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:18013/predict")
    ap.add_argument("--img", required=True)
    ap.add_argument("--stages", default="1:60,2:60,4:60,8:60,16:60",
                    help="并发:秒数，逗号分隔")
    ap.add_argument("--idle-tail", type=int, default=45, help="收尾空载秒数，让面板看到回落")
    ap.add_argument("--out", default="results/loadgen.jsonl")
    a = ap.parse_args()

    blob = open(a.img, "rb").read()
    rec, lock = [], threading.Lock()
    threads, stops = [], []
    marks = []

    def spawn(n):
        for _ in range(n):
            st = threading.Event()
            th = threading.Thread(target=worker, args=(a.url, blob, st, rec, lock), daemon=True)
            th.start(); threads.append(th); stops.append(st)

    t0 = time.time()
    cur = 0
    for stage in a.stages.split(","):
        c, secs = stage.split(":")
        c, secs = int(c), int(secs)
        if c > cur:
            spawn(c - cur)
        elif c < cur:
            for st in stops[c:]:
                st.set()
            del stops[c:]
        cur = c
        marks.append(dict(wall=time.strftime("%H:%M:%S"), t=round(time.time() - t0, 1), concurrency=c))
        print("[%s] T+%5.0fs  并发 -> %d" % (time.strftime("%H:%M:%S"), time.time() - t0, c), flush=True)
        end = time.time() + secs
        while time.time() < end:
            time.sleep(1)
            with lock:
                win = [r for r in rec if r[0] > time.time() - 1]
            ok = sum(1 for r in win if r[1])
            lat = sorted(r[2] for r in win if r[1])
            p99 = lat[int(len(lat) * 0.99)] if lat else float("nan")
            print("   t=%5.0fs conc=%2d qps=%3d p99=%.3fs" %
                  (time.time() - t0, c, ok, p99), flush=True)

    for st in stops:
        st.set()
    marks.append(dict(wall=time.strftime("%H:%M:%S"), t=round(time.time() - t0, 1), concurrency=0))
    print("[%s] 停止加压，空载 %ds" % (time.strftime("%H:%M:%S"), a.idle_tail), flush=True)
    time.sleep(a.idle_tail)

    ok = sum(1 for r in rec if r[1]); err = len(rec) - ok
    lat = sorted(r[2] for r in rec if r[1])
    summ = dict(total=len(rec), ok=ok, err=err,
                p50=lat[len(lat) // 2] if lat else None,
                p99=lat[int(len(lat) * 0.99)] if lat else None,
                wall_start=t0, wall_end=time.time(), marks=marks)
    with open(a.out, "w") as f:
        f.write(json.dumps(summ, ensure_ascii=False) + "\n")
        for w, o, d in rec:
            f.write(json.dumps(dict(wall=w, ok=o, sec=d)) + "\n")
    print("LOADGEN_DONE", json.dumps({k: summ[k] for k in ('total','ok','err','p50','p99')}), flush=True)


if __name__ == "__main__":
    main()
