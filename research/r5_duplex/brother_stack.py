import sys, os, time, json, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import duplex
from PIL import Image
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
BR = _os.environ.get('SCANPP_BROTHER_RAW', _WORK + '/brother/raw') + '/'
ld = lambda i: np.ascontiguousarray(np.array(Image.open(f'{BR}page{i:03d}.tif').convert('RGB')))
pages = [ld(i) for i in range(1, 23)]
t = time.perf_counter(); maps = [duplex.ink_maps(p, 300.) for p in pages]; t_maps = time.perf_counter() - t
t = time.perf_counter(); sheets = duplex.pair_sheets(maps); t_pair = time.perf_counter() - t
print('maps %.0f ms/page  pairing 21 adjacent pairs %.2f s' % (t_maps / 22 * 1000, t_pair))
rows = []
os.makedirs('work/brother_out', exist_ok=True)
for i, j, f in sheets:
    for tgt, srcp, M in ((j, i, f['reg'].M_ab), (i, j, duplex._inv(f['reg'].M_ab))):
        reg = duplex.PairReg(M_ab=M, dpi_A=300., dpi_B=300.)
        out = duplex._clean.clean_page(pages[tgt], 300.)
        t = time.perf_counter(); s = duplex.bleed_suppression(pages[srcp], pages[tgt], reg); ts = time.perf_counter() - t
        new = duplex.apply_suppression(out, s)
        ink = (out.min(-1) < 250); rem = ink & (new.min(-1) >= 250)
        rows.append(dict(page=tgt + 1, source=srcp + 1, ncc=f['ncc'], ga=f['ga'], gb=f['gb'], ink=int(ink.sum()), removed=int(rem.sum()), t_supp=ts))
        print(rows[-1])
        if rem.sum() > 0: np.save(f'work/brother_out/p{tgt+1:03d}_out.npy', out); np.save(f'work/brother_out/p{tgt+1:03d}_new.npy', new)
json.dump(dict(rows=rows, sheets=[(i + 1, j + 1) for i, j, f in sheets], t_maps_ms=t_maps / 22 * 1000, t_pair=t_pair), open('brother_stack.json', 'w'), indent=1)
print('accepted sheets (1-based pages):', [(i + 1, j + 1) for i, j, f in sheets])
