import json, numpy as np
R = json.load(open('work/fine.json'))
def g(d): return 0 if d['border'] else d['peak']-d['second']
print('kind n | ncc pct 10/50/90 | agree med')
for kind in ['adj0','adj1','far','cross']:
    rs=[r for r in R if r['kind']==kind and r.get('ncc') is not None]
    n=np.array([r['ncc'] for r in rs]); ag=[r['agree_in'] for r in rs if r.get('agree_in') is not None]
    print(kind,len(rs),np.percentile(n,[10,50,90,100]).round(2),'agree med',np.median(ag).round(3) if ag else None,'| ncc>=.15:',(n>=.15).sum(),'>=.2',(n>=.2).sum(),'>=.25',(n>=.25).sum(),'>=.3',(n>=.3).sum())
for kind in ['far','cross']:
    print(kind,'top ncc:',sorted([(round(r['ncc'],2),round(r['rho'] or 0,2),round(max(r['ga'],r['gb']),2),(r['agree_in'] and round(r['agree_in'],2)),r['dA']['ink_px'],r['dB']['ink_px']) for r in R if r['kind']==kind and r.get('ncc') is not None],reverse=True)[:12])
