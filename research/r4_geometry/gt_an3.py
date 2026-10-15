import json,numpy as np
R=json.load(open('gt_results.json'))
a=np.array([r['a'] for r in R])
E=lambda n:np.array([r['res'][n][0] for r in R])
Cf=lambda n:np.array([r['res'][n][1] for r in R])
lsd,proj,hough,cc,grad=E('lsd'),E('proj'),E('hough'),E('cc_rect'),E('grad')
def stats(est,acc,name):
    e=np.abs(est-a)[acc]
    print(f"{name:40s} accept {acc.mean()*100:5.1f}%  mean {e.mean():.3f} p95 {np.percentile(e,95):.3f} max {e.max():.2f} fail>0.3 {(e>0.3).mean()*100:.1f}%")
stats(lsd,np.ones(len(a),bool),'lsd all')
for t in (0.2,0.3,0.4,0.5): stats(lsd,Cf('lsd')>=t,f'lsd conf>={t}')
ag=np.abs(lsd-proj)<0.3;stats((lsd+proj)/2,ag,'lsd~proj <0.3 mean');stats(lsd,ag,'lsd~proj <0.3 use lsd')
ag=np.abs(lsd-proj)<0.2;stats((lsd+proj)/2,ag,'lsd~proj <0.2 mean')
ag=np.abs(lsd-proj)<0.3;stats(proj,ag,'lsd~proj <0.3 use proj')
# voting: methods list; find best cluster
def vote(names,tol=0.3,need=2):
    M=np.array([E(n) for n in names]).T;out=np.zeros(len(a));ok=np.zeros(len(a),bool)
    for k in range(len(a)):
        best=None
        for j in range(M.shape[1]):
            s=np.abs(M[k]-M[k,j])<tol
            if best is None or s.sum()>best.sum() or (s.sum()==best.sum() and False): best=s
        if best.sum()>=need: out[k]=M[k,best].mean();ok[k]=True
    return out,ok
for names in (['lsd','proj','cc_rect','grad'],['lsd','proj','hough','cc_rect','grad']):
    for need in (2,3):
        o,k=vote(names,0.3,need);stats(o,k,f'vote {len(names)} need{need}')
# weighted: median of lsd,proj,grad,cc
med=np.median(np.array([lsd,proj,grad,cc]),0);stats(med,np.ones(len(a),bool),'median(lsd,proj,grad,cc)')
