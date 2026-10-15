import os
os.environ["OMP_NUM_THREADS"]="1"
import json,cv2,numpy as np,random
from multiprocessing import Pool
import geometry as G
from fullres import get_full
from sheet import sheet
cv2.setNumThreads(1)
idx=json.load(open('index.json'));E=json.load(open('edge_eval.json'));F=json.load(open('final_survey.json'))
lab=json.load(open('labels_blank.json'))['labels']
blank=np.array([lab.get(str(i),{}).get('label')=='BLANK' for i in range(len(idx))])
ang=np.array([r['ang'] for r in F]);conf=np.array([r['conf'] for r in F])
nb=~blank
coil=np.array([('coil_L' in r['info']) or ('coil_R' in r['info']) for r in E])
strip={s:np.array([r['info']['strip_'+s] for r in E]) for s in 'LRTB'}
# --- 60-page sample: 48 most skewed + 12 random non-blank
random.seed(4)
order=[i for i in np.argsort(-np.abs(ang)) if nb[i] and conf[i]>=0.6]
samp=order[:48]+random.sample([i for i in range(len(idx)) if nb[i] and i not in order[:48]],12)
mm=lambda inch:inch*25.4
print('SAMPLE of 60 (48 most skewed + 12 random): fraction with strip > 2 mm by edge (non-coil strips only; coil sides excluded)')
for s in 'LRTB':
    v=[(strip[s][i]*25.4>2) for i in samp if not (('coil_'+s) in E[i]['info'])]
    print(' ',s,'n=%d'%len(v),'%.1f%%'%(np.mean(v)*100))
print('  coil on a side:',sum(coil[i] for i in samp),'of 60')
anystrip=[max(strip[s][i] for s in 'LRTB') for i in samp]
print('  any non-coil strip >2mm: %.1f%%  >1mm: %.1f%%'%(np.mean([(max(strip[s][i] for s in 'LRTB if not False'.split()[0]) if False else 0) for i in samp])*0,0))
# correlation of strip depth with |angle| (reliable nonblank, non-coil)
k=np.array([nb[i] and conf[i]>=0.6 and not coil[i] for i in range(len(idx))])
mx=np.max([strip[s] for s in 'LRTB'],0)
print('corr(|angle|, max strip depth) over',k.sum(),'non-coil reliable pages: %.2f'%np.corrcoef(np.abs(ang[k]),mx[k])[0,1])
for lo,hi in ((0,0.25),(0.25,0.5),(0.5,1),(1,5)):
    m=k&(np.abs(ang)>=lo)&(np.abs(ang)<hi);print('  |a| in [%.2f,%.2f): n=%d  pages with strip>2mm any side: %.1f%%  median max strip %.2f mm'%(lo,hi,m.sum(),(mx[m]*25.4>2).mean()*100,np.median(mx[m])*25.4))
json.dump([int(i) for i in samp],open('edge_sample60.json','w'))
