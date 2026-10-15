import json,cv2,numpy as np
from methods import ink_map
from sheet import sheet
idx=json.load(open('index.json'));ink=json.load(open('ink_raw.json'))
fc=np.array([x['f0.1'] for x in ink]);ff=np.array([x['ffull0.1'] for x in ink])
sel=sorted(set(np.argsort(fc)[:300])|set(np.argsort(ff)[:300]),key=lambda i:fc[i])
json.dump([int(i) for i in sel],open('label_set.json','w'));print(len(sel))
def boosted(i):
    g=cv2.imread(idx[i]['fn'],0);k=ink_map(g,100)
    return (255*(1-np.clip(k/0.25,0,1))).astype(np.uint8)
per=24
for s in range(0,len(sel),per):
    ch=sel[s:s+per]
    sheet([(boosted(i),f"{i} c{fc[i]*1e4:.1f} f{ff[i]*1e4:.1f}") for i in ch],cols=6,tw=260,out=f'lab_{s//per:02d}.png')
