import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import duplex
from PIL import Image
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
BR = _os.environ.get('SCANPP_BROTHER_RAW', _WORK + '/brother/raw') + '/'
ld = lambda i: np.ascontiguousarray(np.array(Image.open(f'{BR}page{i:03d}.tif').convert('RGB')))
maps = [duplex.ink_maps(ld(i), 300.) for i in range(1, 23)]
rows = []
for i in range(21):
    f = duplex.register_pair(maps[i], maps[i + 1]); ok = duplex.accept(f, maps[i], maps[i + 1])
    rows.append(dict(a=i + 1, b=i + 2, true_sheet=bool(i % 2 == 0), gap_a=round(f['ga'], 3), gap_b=round(f['gb'], 3), ncc=None if f.get('ncc') is None else round(f['ncc'], 3),
                     agree_in=None if f.get('agree_in') is None else round(f['agree_in'], 3), chroma_px=(duplex.chroma_px(maps[i]), duplex.chroma_px(maps[i + 1])), accept=bool(ok)))
    print(rows[-1])
json.dump(rows, open('brother_gate_table.json', 'w'), indent=1)
