import json,numpy as np
R=json.load(open('gt_results.json'));A=np.load('A.npy')  # A is from the OLD survey (no lsd)
# lsd on unrotated pages not computed in A; compute quickly
import cv2
from methods import est_lsd
idx=json.load(open('index.json'))
cand=sorted(set(r['i'] for r in R))
base={i:est_lsd(cv2.imread(idx[i]['fn'],0),100)[0] for i in cand}
rel=np.array([r['res']['lsd'][0]-r['a']-base[r['i']] for r in R])
print('lsd baseline of GT pages: |a| mean',np.mean(np.abs(list(base.values()))),'max',np.max(np.abs(list(base.values()))))
print('lsd relative error: mean %.3f p95 %.3f max %.3f fail>0.3 %.1f%% >0.15 %.1f%%'%(np.abs(rel).mean(),np.percentile(np.abs(rel),95),np.abs(rel).max(),(np.abs(rel)>0.3).mean()*100,(np.abs(rel)>0.15).mean()*100))
