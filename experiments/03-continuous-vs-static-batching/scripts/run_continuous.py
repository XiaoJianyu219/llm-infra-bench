import asyncio, json, sys, time
import aiohttp
from workload import make_requests, PROMPT

MODEL = "/root/autodl-tmp/models/Qwen3-VL-8B-Instruct"
URL = "http://localhost:8000/v1/chat/completions"

async def one(sess, req, t0, out):
    await asyncio.sleep(max(0, req["arrive"] - (time.perf_counter() - t0)))
    payload = {"model": MODEL,
               "messages": [{"role": "user", "content": PROMPT}],
               "max_tokens": req["out_len"], "temperature": 0,
               "ignore_eos": True}
    async with sess.post(URL, json=payload) as r:
        await r.json()
    done = time.perf_counter() - t0
    out.append({"id": req["id"], "arrive": req["arrive"], "done": done,
                "latency": done - req["arrive"], "out_len": req["out_len"]})

async def main(mode):
    reqs = make_requests(mode=mode)
    out = []
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn,
              timeout=aiohttp.ClientTimeout(total=1800)) as sess:
        t0 = time.perf_counter()
        await asyncio.gather(*[one(sess, r, t0, out) for r in reqs])
        makespan = time.perf_counter() - t0
    json.dump({"strategy": "continuous", "mode": mode,
               "makespan": makespan, "records": out},
              open(f"/root/autodl-tmp/bench/day03/continuous_{mode}.json", "w"))
    print(f"continuous / {mode}: makespan {makespan:.1f}s, "
          f"吞吐 {len(reqs)/makespan:.2f} req/s")

asyncio.run(main(sys.argv[1]))