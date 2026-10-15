import json,numpy as np
R=json.load(open('gt_results.json'))
N=list(R[0]['res'])
a=np.array([r['a'] for r in R])
for n in N:
    e=np.array([r['res'][n][0]-r['a'] for r in R]);ae=np.abs(e)
    print(f"{n:8s} mean {ae.mean():.3f} med {np.median(ae):.3f} p95 {np.percentile(ae,95):.3f} max {ae.max():.2f} fail>0.3 {(ae>0.3).mean()*100:.1f}%  bias {e.mean():+.3f} ms {np.median([r['res'][n][2] for r in R]):.0f}")
# combos
def comb(names,how='median'):
    E=np.array([[r['res'][n][0] for n in names] for r in R]);return np.median(E,1)
for names in (['proj','hough','cc_rect','grad'],['proj','hough','grad'],['proj','hough'],['proj','grad'],['hough','grad','cc_rect']):
    e=np.abs(comb(names)-a);print('median of',names,'mean %.3f p95 %.3f max %.2f fail %.1f%%'%(e.mean(),np.percentile(e,95),e.max(),(e>0.3).mean()*100))
# failures by angle
for n in N:
    ae=np.array([abs(r['res'][n][0]-r['a']) for r in R]);print(n,'fail by |a| bins',[round(float((ae[(np.abs(a)>=lo)&(np.abs(a)<hi)]>0.3).mean()*100),1) for lo,hi in [(0,2),(2,4),(4,6),(6,8.1)]])
print('--- relative to method own estimate on unrotated page (cancels page residual skew)')
A=np.load('A.npy');NN=['proj','hough','cc_rect','cc_mom','grad']
for n in N:
    j=NN.index(n)
    e=np.array([r['res'][n][0]-r['a']-A[r['i'],j] for r in R]);ae=np.abs(e)
    print(f"{n:8s} mean {ae.mean():.3f} med {np.median(ae):.3f} p95 {np.percentile(ae,95):.3f} max {ae.max():.2f} fail>0.3 {(ae>0.3).mean()*100:.1f}% >0.15 {(ae>0.15).mean()*100:.1f}%")
# vs consensus-corrected truth
cons=np.median(A[:,:4],1)
for n in N:
    j=NN.index(n);e=np.array([r['res'][n][0]-(r['a']+cons[r['i']]) for r in R]);ae=np.abs(e)
    print(f"{n:8s} vs a+consensus0: mean {ae.mean():.3f} p95 {np.percentile(ae,95):.3f} max {ae.max():.2f} fail>0.3 {(ae>0.3).mean()*100:.1f}%")
# confidence vs error
for n in ['proj','hough','cc_rect']:
    c=np.array([r['res'][n][1] for r in R]);ae=np.array([abs(r['res'][n][0]-r['a']) for r in R])
    for q in (0.2,0.4,0.6):
        m=c<q;print(n,'conf<',q,'n',m.sum(),'fail',(ae[m]>0.3).sum(),' | conf>=',q,'n',(~m).sum(),'fail',(ae[~m]>0.3).sum())
