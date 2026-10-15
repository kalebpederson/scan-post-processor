import json,numpy as np
R=json.load(open('edges_raw.json'));idx=json.load(open('index.json'))
ink=json.load(open('ink_raw.json'));fc=np.array([x['f0.1'] for x in ink]);blank=fc<5e-4
mm=lambda px:px/100*25.4
for S in 'LRTB':
    run=np.array([r['run_'+S] for r in R]);cov=np.array([r['cov_'+S] for r in R])
    print(S,'frac pages max-thickness>0.5mm(2px):%.3f >1mm:%.3f >2mm(8px):%.3f >4mm(16px):%.3f ; coverage>0.5: %.3f'%tuple(float((run>t).mean()) for t in (2,4,8,16))+(float((cov>0.5).mean()),) if False else '', end='')
    print(S,[round(float((run>t).mean()),3) for t in (2,4,8,16)],'cov>0.5',round(float((cov>0.5).mean()),3),'cov>0.9',round(float((cov>0.9).mean()),3))
anyrun=np.max([[r['run_'+s] for s in 'LRTB'] for r in R],1)
print('any side >2mm:',(anyrun>7.9).mean(),'>1mm',(anyrun>3.9).mean(),'>4mm',(anyrun>15.7).mean(), ' blank pages any>2mm',(anyrun[blank]>7.9).mean(),'nonblank',(anyrun[~blank]>7.9).mean())
# distributions of thick strips
for S in 'LRTB':
    run=np.array([r['run_'+S] for r in R]);print(S,'max run px pcts 50/90/99/max',np.round(np.percentile(run,[50,90,99,100]),1))
nh=np.array([len(r['holes']['L']) for r in R]);nr=np.array([len(r['holes']['R']) for r in R])
print('pages with holes L>=1',(nh>=1).sum(),'L>=5',(nh>=5).sum(),'R>=1',(nr>=1).sum(),'R>=5',(nr>=5).sum())
print('hole count hist L',np.bincount(np.minimum(nh,12)),'R',np.bincount(np.minimum(nr,12)))
