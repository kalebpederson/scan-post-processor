import numpy as np
from detlib import *
L = lambda i: np.load(f'work/brother50/{i:03d}.npy').astype(np.float32)
for i in range(1, 23):
    for j in (i+1,):
        if j > 22: continue
        a, b = pair_scores(L(i), L(j))
        tag = 'SHEET' if i % 2 == 1 else 'cross'
        f = lambda d: 'z%.1f pk%.2f gap%.2f (%d,%d)%s' % (d['z'], d['peak'], d['peak']-d['second'], d['dx'], d['dy'], 'B' if d['border'] else '')
        print(i, j, tag, 'a:', f(a), '| b:', f(b))
