import json,numpy as np
R=json.load(open('gt_results.json'));B=json.load(open('gt_base.json'))
a=np.array([r['a'] for r in R])
E=lambda n:np.array([r['res'][n][0] for r in R])
bs=lambda n:np.array([B[str(r['i'])][n] for r in R])
def stats(est,acc,name,rel=False,n=None):
    ref=a+(bs(n) if rel else 0)
    e=np.abs(est-ref)[acc]
    print(f"{name:34s} {'rel' if rel else 'abs'} accept {acc.mean()*100:5.1f}%  mean {e.mean():.3f} p95 {np.percentile(e,95):.3f} max {e.max():.2f} fail>0.3 {(e>0.3).mean()*100:.1f}% (n={int((e>0.3).sum())})")
lsd,proj,cc,grad=E('lsd'),E('proj'),E('cc_rect'),E('grad')
one=np.ones(len(a),bool)
for tol in (0.2,0.3,0.4):
    ag=np.abs(lsd-proj)<tol
    stats((lsd+proj)/2,ag,f'mean(lsd,proj) agree<{tol}')
    stats(proj,ag,f'proj agree<{tol}');stats(lsd,ag,f'lsd agree<{tol}')
# three way: lsd,proj,cc
M=np.array([lsd,proj,cc]).T
def vote(M,tol,need):
    out=np.zeros(len(M));ok=np.zeros(len(M),bool)
    for k in range(len(M)):
        best=None
        for j in range(M.shape[1]):
            s=np.abs(M[k]-M[k,j])<tol
            if best is None or s.sum()>best.sum(): best=s
        if best.sum()>=need: out[k]=M[k,best].mean();ok[k]=True
    return out,ok
for tol in (0.25,0.35):
    o,k=vote(M,tol,2);stats(o,k,f'vote(lsd,proj,cc) tol{tol} need2')
    o,k=vote(np.array([lsd,proj,cc,grad]).T,tol,2);stats(o,k,f'vote4 tol{tol} need2')
    o,k=vote(np.array([lsd,proj,cc,grad]).T,tol,3);stats(o,k,f'vote4 tol{tol} need3')
print('--- self-consistency of ensembles')
bl,bp,bc,bg=bs('lsd'),bs('proj'),bs('cc_rect'),bs('grad')
def rel(est,base,acc,name):
    e=np.abs(est-base-a)[acc]
    print(f"{name:34s} accept {acc.mean()*100:5.1f}% mean {e.mean():.3f} p95 {np.percentile(e,95):.3f} max {e.max():.2f} fail>0.3 {(e>0.3).mean()*100:.1f}% (n={int((e>0.3).sum())}) >0.15 {(e>0.15).mean()*100:.1f}%")
rel(lsd,bl,one,'lsd');rel(proj,bp,one,'proj');rel((lsd+proj)/2,(bl+bp)/2,one,'mean(lsd,proj)')
rel(np.median([lsd,proj,cc],0),np.median([bl,bp,bc],0),one,'median(lsd,proj,cc)')
ag=np.abs(lsd-proj)<0.3;rel((lsd+proj)/2,(bl+bp)/2,ag,'mean(lsd,proj) | agree<0.3')
