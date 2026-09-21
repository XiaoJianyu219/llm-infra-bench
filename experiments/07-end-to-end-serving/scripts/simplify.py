"""Day 5 replacement algorithm, using structural matching to omit private class names."""
def simplify(net, verbose=False):
    seq=net.model
    idx=[i for i,mod in enumerate(seq) if hasattr(mod,'mwas') and hasattr(mod.mwas,'adapter')]
    for i in idx:
        mod=seq[i]; ad=mod.mwas.adapter
        for attr in ('i','f'): setattr(ad,attr,getattr(mod,attr))
        ad.type='equivalent_adapter'
        ad.np=sum(p.numel() for p in ad.parameters())
        seq[i]=ad
    return len(idx)
