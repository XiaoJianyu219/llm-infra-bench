import random

PROMPT = "请详细分析遥感图像中小目标检测的技术难点，" * 20  # 约 500 token

def make_requests(n=200, rate=4.0, mode="fixed", seed=42):
    """泊松到达。mode: fixed / varlen"""
    rng = random.Random(seed)
    reqs, t = [], 0.0
    for i in range(n):
        t += rng.expovariate(rate)
        out = 128 if mode == "fixed" else rng.choice([32, 64, 128, 256, 512])
        reqs.append({"id": i, "arrive": t, "out_len": out})
    return reqs