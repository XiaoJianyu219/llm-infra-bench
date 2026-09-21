# -*- coding: utf-8 -*-
"""把 DeFE_MWAS 等价替换为它内部的 adapter。

依据(三条独立证据均已验证):
  1. MWAS.forward 里 center_density = pooled[0,0]，pooled 是 3 维，
     切完是 1 维 → numel = nW，永远 != nH*nW → 掩码恒被 zeros 覆盖;
  2. 运行时探针: 真正执行掩码窗口注意力的 0/4;
  3. prove_equiv.py 逐位比对 5 组输入，最大绝对误差 0.000e+00。
掩码分支不执行时，MWAS 的 pad/unfold/permute/view 尾部完全逆向自身，
故 MWAS(x, density) ≡ adapter(x)，DeFE 算出的密度图只喂给死分支。
"""

TARGET = 'DeFE_MWAS'


def simplify(net, verbose=True):
    seq = net.model                      # ultralytics DetectionModel 的 nn.Sequential
    idx = [i for i, mod in enumerate(seq) if type(mod).__name__ == TARGET]
    for i in idx:
        mod = seq[i]
        ad = mod.mwas.adapter
        for attr in ('i', 'f'):          # ultralytics 前向循环依赖这两个属性
            setattr(ad, attr, getattr(mod, attr))
        ad.type = f'{TARGET}->adapter'
        ad.np = sum(p.numel() for p in ad.parameters())
        seq[i] = ad
        if verbose:
            print(f'  layer {i:>2}: {TARGET} -> adapter')
    return len(idx)
