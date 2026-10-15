import sys, json, numpy as np, cv2
from detlib import *
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
idx = json.load(open(_ROOT + '/research/r4_geometry/index.json'))
R4=_ROOT + '/research/r4_geometry/'
def show(i, j, out):
    A = cv2.imread(R4+idx[i]['fn']); B = cv2.imread(R4+idx[j]['fn'])[:, ::-1]
    h=min(A.shape[0],B.shape[0]); w=min(A.shape[1],B.shape[1])
    fa = load(i)[:h//2*1,:]  # unused
    cv2.imwrite(out, np.hstack([A[:h,:w], np.ascontiguousarray(B[:h,:w])]))
if __name__=='__main__':
    show(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
