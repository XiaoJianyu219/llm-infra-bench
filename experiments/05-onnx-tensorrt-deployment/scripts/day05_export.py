# -*- coding: utf-8 -*-
import os, sys, glob, time, torch
from ultralytics import YOLO
from simplify import simplify

W      = '/root/autodl-tmp/sod/best.pt'
IMGSZ  = 1024
DEV    = 'cuda:0'
IMGDIR = '/root/autodl-tmp/sod/eval800/images'     # 没有就自动退回随机张量

assert os.path.exists(W), f'找不到权重: {W}'


def head(t):
    return t[0] if isinstance(t, (list, tuple)) else t


@torch.no_grad()
def fwd(net, x):
    return head(net(x)).float().clone()


@torch.no_grad()
def bench(net, x, warm=10, rep=50):
    for _ in range(warm):
        net(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        net(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / rep * 1000


def load_inputs(k=5):
    fs = sorted(glob.glob(os.path.join(IMGDIR, '*.jpg')))[:k] if os.path.isdir(IMGDIR) else []
    if fs:
        import cv2
        xs = []
        for f in fs:
            im = cv2.resize(cv2.imread(f), (IMGSZ, IMGSZ))
            im = im[:, :, ::-1].transpose(2, 0, 1).copy()
            xs.append(torch.from_numpy(im).float().div(255).unsqueeze(0).to(DEV))
        print(f'输入: {len(xs)} 张真实图片')
        return xs
    torch.manual_seed(0)
    print('输入: 随机张量 (未找到 eval800 图片, 数值对齐结论偏保守)')
    return [torch.randn(1, 3, IMGSZ, IMGSZ, device=DEV) for _ in range(k)]


m   = YOLO(W)
net = m.model.to(DEV).eval().float()
for p in net.parameters():
    p.requires_grad_(False)

xs  = load_inputs()
p0  = sum(p.numel() for p in net.parameters())
ref = [fwd(net, x) for x in xs]
torch.cuda.reset_peak_memory_stats()
t0   = bench(net, xs[0])
mem0 = torch.cuda.max_memory_allocated() / 2 ** 20

print('\n===== 1. 等价简化 =====')
n = simplify(net)
print(f'共替换 {n} 个 DeFE_MWAS')
assert n == 4, f'期望 4 个, 实际 {n} 个'

p1   = sum(p.numel() for p in net.parameters())
errs = [(fwd(net, x) - r).abs().max().item() for x, r in zip(xs, ref)]
print('逐位一致性: ' + '  '.join(f'{e:.3e}' for e in errs))
assert max(errs) == 0.0, '简化后不是恒等, 停止'

torch.cuda.reset_peak_memory_stats()
t1   = bench(net, xs[0])
mem1 = torch.cuda.max_memory_allocated() / 2 ** 20

print(f'参数量   {p0:,} -> {p1:,}   (-{p0 - p1:,}, -{100 * (p0 - p1) / p0:.2f}%)')
print(f'延迟     {t0:.2f} ms -> {t1:.2f} ms   (加速 {t0 / t1:.3f}x)')
print(f'峰值显存 {mem0:.0f} MB -> {mem1:.0f} MB')

print('\n===== 2. 导出 ONNX =====')
try:
    onnx_path = m.export(format='onnx', imgsz=IMGSZ, opset=17,
                         simplify=True, dynamic=False, device=0)
except Exception as e:
    print('simplify=True 失败, 退回 False:', type(e).__name__, str(e)[:150])
    onnx_path = m.export(format='onnx', imgsz=IMGSZ, opset=17,
                         simplify=False, dynamic=False, device=0)
print('ONNX:', onnx_path)

import onnx
g   = onnx.load(onnx_path)
ops = {}
for nd in g.graph.node:
    ops[nd.op_type] = ops.get(nd.op_type, 0) + 1
bad = {k: ops[k] for k in ('NonZero', 'Where', 'Loop', 'If', 'Range') if k in ops}
print(f'节点数 {sum(ops.values())}, 算子种类 {len(ops)}')
print('数据依赖型算子:', bad if bad else '无 ✓')

print('\n===== 3. ONNX 数值对齐 =====')
import onnxruntime as ort
provs = ort.get_available_providers()
use   = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in provs else ['CPUExecutionProvider']
print('provider:', use[0], '(CPU 会慢, 只跑 2 个样本)')
sess  = ort.InferenceSession(onnx_path, providers=use)
iname = sess.get_inputs()[0].name
rows  = []
for i, x in enumerate(xs[:2]):
    o = torch.from_numpy(sess.run(None, {iname: x.cpu().numpy()})[0]).to(DEV).float()
    d = (o - ref[i]).abs()
    rel = (d.max() / ref[i].abs().max()).item()
    print(f'  样本{i}: shape={tuple(o.shape)}  最大绝对误差={d.max():.3e}  '
          f'平均={d.mean():.3e}  相对={rel:.3e}')
    rows.append((d.max().item(), d.mean().item(), rel))

with open('day05_metrics.txt', 'w') as f:
    f.write(f'params_before\t{p0}\nparams_after\t{p1}\n')
    f.write(f'latency_ms_before\t{t0:.3f}\nlatency_ms_after\t{t1:.3f}\n')
    f.write(f'mem_mb_before\t{mem0:.1f}\nmem_mb_after\t{mem1:.1f}\n')
    f.write(f'bitexact_max_err\t{max(errs):.3e}\n')
    f.write(f'onnx_path\t{onnx_path}\nonnx_nodes\t{sum(ops.values())}\n')
    f.write(f'onnx_dyn_ops\t{bad}\n')
    for i, (a, b, c) in enumerate(rows):
        f.write(f'onnx_vs_torch_{i}\tmax={a:.3e}\tmean={b:.3e}\trel={c:.3e}\n')
print('\n指标已写入 day05_metrics.txt')
