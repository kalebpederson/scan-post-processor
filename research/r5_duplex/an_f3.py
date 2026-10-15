import json, numpy as np
R = json.load(open('work/fine.json'))
def srcc(r):   # chroma px of the source of the best direction ('a': A receives from B -> source B)
    return r['dB']['chroma_px'] if r['best']=='a' else r['dA']['chroma_px']
def rule(r, Cmin=400, n2=0.2, n1=0.5, g2=0.05, g1=0.06, ag=0.12):
    if r.get('ncc') is None: return False
    two = r.get('agree_in') is not None and r['ga']>=g2 and r['gb']>=g2 and r['agree_in']<=ag and r['ncc']>=n2
    one = max(r['ga'],r['gb'])>=g1 and r['ncc']>=n1
    return (two or one) and srcc(r)>=Cmin
tot={'adj0':904,'adj1':755,'far':5541,'cross':38144}
for kind in tot:
    rs=[r for r in R if r['kind']==kind]
    print(kind,tot[kind],[ (C,sum(rule(r,Cmin=C) for r in rs)) for C in (0,200,400,800,1600)])
for kind in ('far','cross'):
    for r in R:
        if r['kind']==kind and rule(r,Cmin=0): print(kind,r['i'],r['j'],'g %.2f %.2f ncc %.2f agree %s srcchroma %d'%(r['ga'],r['gb'],r['ncc'],r['agree_in'],srcc(r)))
