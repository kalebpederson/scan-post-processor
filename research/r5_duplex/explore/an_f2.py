import json, numpy as np, truth
R = json.load(open('work/fine.json'))
def T(r, g0=0.05, ag=0.12): return r.get('agree_in') is not None and r['ga']>=g0 and r['gb']>=g0 and r['agree_in']<=ag
def O(r, g0=0.15, n0=0.35): return r.get('ncc') is not None and max(r['ga'],r['gb'])>=g0 and r['ncc']>=n0
tot={'adj0':904,'adj1':755,'far':5541,'cross':38144}
for kind in tot:
    rs=[r for r in R if r['kind']==kind]
    print(kind,tot[kind],'two-way',sum(T(r) for r in rs),' two-way(g.08)',sum(T(r,0.08) for r in rs),'one-way',sum(O(r) for r in rs),'either',sum(T(r) or O(r) for r in rs))
for kind in ('far','cross'):
    for r in R:
        if r['kind']==kind and (T(r) or O(r)): print(kind,'T' if T(r) else '-', 'O' if O(r) else '-', r['i'],r['j'],'g %.2f %.2f ncc %.2f agree %s ink %d %d'%(r['ga'],r['gb'],r['ncc'],r['agree_in'],r['dA']['ink_px'],r['dB']['ink_px']))
