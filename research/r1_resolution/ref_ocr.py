import sys, json, cv2
from multiprocessing import Pool
from lib import *
def job(a):
    d,psm=a
    g=cv2.imread(f'{WORK}/gray_{d}.png',0)
    t=time.time(); w=ocr(g,d,psm); 
    json.dump(w,open(f'{WORK}/refocr_{d}_psm{psm}.json','w'))
    return d,psm,len(w),time.time()-t
if __name__=='__main__':
    with Pool(4) as p:
        for r in p.imap_unordered(job,[(600,3),(600,11),(1200,3),(1200,11)]): print(r,flush=True)
