import json,numpy as np
R=json.load(open('gt_results.json'))
N=list(R[0]['res']);print(len(R),N)
a=np.array([r['a'] for r in R]);big=np.arange(len(R))<180
def rep(mask,title):
    print('==',title,mask.sum())
    for n in N:
        e=np.array([r['res'][n][0]-r['a'] for r in R])[mask];ae=np.abs(e)
        print(f"{n:8s} mean {ae.mean():.3f} med {np.median(ae):.3f} p95 {np.percentile(ae,95):.3f} max {ae.max():.2f} fail>0.3 {(ae>0.3).mean()*100:.1f}% bias {e.mean():+.3f} ms {np.median([r['res'][n][2] for r in R]):.0f}")
rep(big,'+-8 deg (180)');rep(~big,'+-2.5 deg (120)');rep(np.ones(len(R),bool),'all')
