import json, sys, time, torch
from transformers import AutoTokenizer, AutoModelForImageTextToText
from workload import make_requests, PROMPT

M = "/root/autodl-tmp/models/Qwen3-VL-8B-Instruct"
BATCH, WINDOW = 16, 0.05          # 攒够 16 个，或等待 50 ms

tok = AutoTokenizer.from_pretrained(M)
tok.padding_side = "left"
model = AutoModelForImageTextToText.from_pretrained(M, dtype=torch.bfloat16).to("cuda").eval()

@torch.no_grad()
def run_batch(k, max_new):
    """真实跑一批 k 个请求、每个生成 max_new token，返回耗时"""
    enc = tok([PROMPT] * k, return_tensors="pt", padding=True).to("cuda")
    torch.cuda.synchronize(); t = time.perf_counter()
    model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                   min_new_tokens=max_new, pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    return time.perf_counter() - t

def main(mode):
    reqs = make_requests(mode=mode)
    run_batch(2, 8)                      # warmup
    clock, i, out = 0.0, 0, []
    while i < len(reqs):
        batch = reqs[i:i + BATCH]
        # 批开始时刻 = max(最后一个成员到达时刻 + 窗口, 上一批结束)
        start = max(batch[-1]["arrive"] + WINDOW, clock)
        max_new = max(r["out_len"] for r in batch)   # 整批等最长的
        dur = run_batch(len(batch), max_new)
        clock = start + dur
        for r in batch:
            out.append({"id": r["id"], "arrive": r["arrive"], "done": clock,
                        "latency": clock - r["arrive"], "out_len": r["out_len"]})
        i += BATCH
        print(f"batch {i//BATCH}: k={len(batch)} max_new={max_new} dur={dur:.2f}s")
    makespan = clock
    json.dump({"strategy": "static", "mode": mode, "batch": BATCH,
               "window": WINDOW, "makespan": makespan, "records": out},
              open(f"/root/autodl-tmp/bench/day03/static_{mode}.json", "w"))
    print(f"static / {mode}: makespan {makespan:.1f}s, 吞吐 {len(reqs)/makespan:.2f} req/s")

main(sys.argv[1])