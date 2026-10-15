import cv2,numpy as np,time,json
from methods import *
from methods import _rot
idx=json.load(open('index.json'))
for k in [(82,1),(70,1),(40,0),(267,30)]:
    m=[x for x in idx if (x['k'],x['page'])==k][0]
    g=cv2.imread(m['fn'],0)
    for n,f in METHODS.items():
        t=time.time();r=f(g,100);print(k,n,np.round(r,3),round((time.time()-t)*1000),'ms')
    # check angle sign with synthetic
    gr=255-_rot(255-g,-3.0)
    print('synthetic +3:',{n:np.round(f(gr,100)[0],2) for n,f in METHODS.items()})
