# -*- coding: utf-8 -*-
import os, glob, time, torch, numpy as np, cv2
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from simplify import simplify

W, IMGSZ, DEV = '/root/autodl-tmp/sod/best.pt', 1024, 'cuda:0'
N_IMG, CONF = 20, 0.25

# ---------- 1. 找图 ----------
cands = []
for root in ('/root/autodl-tmp', '/root/data'):
    if not os.path.isdir(root):
        continue
    for dp, dn, fn in os.walk(root):
        if '/venvs/' in dp or '/site-packages/' in dp or '/.git/' in dp:
            dn[:] = []
            continue
        imgs = [f for f in fn if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        if len(imgs) >= 50:
            cands.append((len(imgs), dp))
cands.sort(reverse=True)
print('===== 候选图像目录 (>=50 张) =====')
for n, d in cands[:10]:
    lbl = d.replace('/images', '/labels')
    nl = len(glob.glob(os.path.join(lbl, '*.txt'))) if os.path.isdir(lbl) else 0
    print(f'  {n:>6} 张  labels={nl:>6}  {d}')
assert cands, '没找到图像目录,请确认上传路径'
IMGDIR = cands[0][1]
print(f'\n使用: {IMGDIR}')

files = sorted(glob.glob(os.path.join(IMGDIR, '*')))
files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))][:N_IMG]

lb = LetterBox((IMGSZ, IMGSZ), auto=False, scaleup=True)
def prep(f):
    im = lb(image=cv2.imread(f))
    im = im[:, :, ::-1].transpose(2, 0, 1).copy()
    return torch.from_numpy(im).float().div(255).unsqueeze(0)

# ---------- 2. 模型 ----------
m   = YOLO(W)
net = m.model.to(DEV).eval().float()
for p in net.parameters(): p.requires_grad_(False)
assert simplify(net, verbose=False) == 4

import onnxruntime as ort
sess  = ort.InferenceSession('/root/autodl-tmp/sod/best.onnx',
                             providers=['CPUExecutionProvider'])
iname = sess.get_inputs()[0].name

# ---------- 3. 对齐 ----------
print(f'\n===== 真实图像对齐 ({len(files)} 张, conf>{CONF}) =====')
tot_keep, box_errs, cls_errs, rank_bad, cls_flip = 0, [], [], 0, 0
for k, f in enumerate(files):
    x = prep(f)
    with torch.no_grad():
        t = net(x.to(DEV))[0].float().cpu()[0]          # (8, 87040)
    o = torch.from_numpy(sess.run(None, {iname: x.numpy()})[0]).float()[0]

    conf_t = t[4:8].max(0).values
    keep   = conf_t > CONF
    nk     = int(keep.sum())
    tot_keep += nk
    if nk == 0:
        print(f'  [{k:>2}] {os.path.basename(f)[:34]:<34} 高分框 0')
        continue

    db = (t[0:4, keep] - o[0:4, keep]).abs()
    dc = (t[4:8, keep] - o[4:8, keep]).abs()
    box_errs.append(db.max().item()); cls_errs.append(dc.max().item())

    # 类别归属是否翻转
    flip = (t[4:8, keep].argmax(0) != o[4:8, keep].argmax(0)).sum().item()
    cls_flip += flip
    # ONNX 侧是否还是同一批框被保留
    rank_bad += int((o[4:8].max(0).values > CONF).sum().item() != nk)

    print(f'  [{k:>2}] {os.path.basename(f)[:34]:<34} 高分框 {nk:>4}  '
          f'box最大{db.max():.3e}px  cls最大{dc.max():.3e}  类别翻转 {flip}')

print(f'\n高分框总数 {tot_keep}')
if box_errs:
    print(f'box  最大 {max(box_errs):.3e} px   中位 {np.median(box_errs):.3e} px')
    print(f'cls  最大 {max(cls_errs):.3e}      中位 {np.median(cls_errs):.3e}')
    print(f'类别归属翻转总数: {cls_flip}')
    print(f'保留框数量不一致的图: {rank_bad}/{len(box_errs)}')
else:
    print('!! 所有图都没有高分框 —— 检查权重/预处理是否匹配')
