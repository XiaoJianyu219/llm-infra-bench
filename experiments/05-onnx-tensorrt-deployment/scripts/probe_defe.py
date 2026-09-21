import torch, torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn.modules import DeFE_MWAS
import ultralytics.nn.modules.block as B

log = []

def patch(mod, idx):
    orig = mod.forward
    def wrapped(x, density):
        H, W = x.shape[2:]          # 注意：adapter 之前的 shape
        return orig(x, density)
    # 直接在 MWAS 内部复算一遍判据
    def probe(x, density):
        xa = mod.adapter(x)
        H, W = xa.shape[2:]
        ws = mod.window_size
        ph = (ws - H % ws) % ws; pw = (ws - W % ws) % ws
        d = F.pad(density, (0, pw, 0, ph)) if (ph or pw) else density
        nH, nW = (H + ph)//ws, (W + pw)//ws
        pooled = F.avg_pool2d(d[0], ws)
        cd = pooled[0, 0]
        ok = cd.numel() == nH * nW
        log.append(f"  MWAS#{idx}: feat={H}x{W} ws={ws} nH,nW=({nH},{nW}) "
                   f"center_density={tuple(cd.shape)} numel={cd.numel()} "
                   f"需要={nH*nW} → {'✓走注意力' if ok else '✗走全零fallback'}")
        return orig(x, density)
    mod.forward = probe

m = YOLO('/root/autodl-tmp/sod/best.pt')
net = m.model.eval()
mwas_list = [mm for mm in net.modules() if type(mm).__name__ == "MWAS"]
print(f"找到 {len(mwas_list)} 个 MWAS")
for i, mod in enumerate(mwas_list):
    patch(mod, i)

with torch.no_grad():
    net(torch.randn(1, 3, 1024, 1024))

print("\n".join(log))
ok_cnt = sum("✓" in s for s in log)
print(f"\n真正执行掩码窗口注意力的: {ok_cnt}/{len(log)}")
