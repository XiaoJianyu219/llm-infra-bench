import torch, copy
from ultralytics import YOLO

m = YOLO('/root/autodl-tmp/sod/best.pt')
net = m.model.eval()

xs = [torch.randn(1, 3, 1024, 1024) for _ in range(5)]
with torch.no_grad():
    ref = [net(x)[0].clone() for x in xs]

# 等价替换：DeFE_MWAS 直接退化成 mwas.adapter
n_patched = 0
for mod in net.modules():
    if type(mod).__name__ == "DeFE_MWAS":
        mod.forward = (lambda ad: (lambda x: ad(x)))(mod.mwas.adapter)
        n_patched += 1
print(f"替换了 {n_patched} 个 DeFE_MWAS")

with torch.no_grad():
    new = [net(x)[0] for x in xs]

for i, (a, b) in enumerate(zip(ref, new)):
    d = (a - b).abs().max().item()
    print(f"  输入{i}: 最大绝对误差 {d:.3e}   {'恒等' if d == 0 else ('数值一致' if d < 1e-5 else '不一致！')}")
