import time, torch
from transformers import AutoTokenizer
try:
    from transformers import AutoModelForImageTextToText as AutoCls
except ImportError:
    from transformers import AutoModelForCausalLM as AutoCls

M = "/root/autodl-tmp/models/Qwen3-VL-8B-Instruct"
tok = AutoTokenizer.from_pretrained(M)
model = AutoCls.from_pretrained(M, dtype=torch.bfloat16).to("cuda").eval()

prompt = "请用中文详细介绍遥感图像小目标检测的数据集构建、标注规范、模型结构、损失函数与评价指标。" * 12
ids = tok(prompt, return_tensors="pt").input_ids.cuda()

@torch.no_grad()
def no_cache(n):
    x = ids.clone(); fwd = 0
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        out = model(input_ids=x, use_cache=False)
        fwd += x.shape[1]
        x = torch.cat([x, out.logits[:, -1:].argmax(-1)], dim=-1)
    torch.cuda.synchronize()
    return time.time() - t0, fwd

@torch.no_grad()
def with_cache(n):
    torch.cuda.synchronize(); t0 = time.time()
    out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    nxt  = out.logits[:, -1:].argmax(-1)
    fwd  = ids.shape[1]
    for _ in range(n - 1):
        out  = model(input_ids=nxt, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt  = out.logits[:, -1:].argmax(-1)
        fwd += 1
    torch.cuda.synchronize()
    return time.time() - t0, fwd

with_cache(4)   # warmup
print(f"prompt = {ids.shape[1]} tokens\n")
print(f"{'生成':>6} {'有cache':>12} {'无cache':>12} {'加速比':>8} {'前向token数比':>14}")
for n in (16, 32, 64):
    tc, fc = with_cache(n)
    tn, fn = no_cache(n)
    print(f"{n:>6} {tc:>10.2f}s {tn:>10.2f}s {tn/tc:>7.2f}× {fn}/{fc} = {fn/fc:>5.1f}×")
