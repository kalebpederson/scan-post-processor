import json, numpy as np
R = json.load(open('work/coarse.json'))
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[3]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
idx = json.load(open(_ROOT + '/research/r4_geometry/index.json'))
def g(d): return 0 if d['border'] else d['peak']-d['second']
rows=[r for r in R if r['kind']=='cross' and max(g(r['a']),g(r['b']))>=0.08]
for r in sorted(rows,key=lambda r:-max(g(r['a']),g(r['b']))):
    a,b=r['a'],r['b']
    print('%.2f %.2f pk %.2f %.2f z %.0f %.0f  shift (%d,%d)(%d,%d) chroma_px A %d B %d ink A %d B %d | %s p%d | %s p%d'%(g(a),g(b),a['peak'],b['peak'],a['z'],b['z'],a['dx'],a['dy'],b['dx'],b['dy'],r['dA']['chroma_px'],r['dB']['chroma_px'],r['dA']['ink_px'],r['dB']['ink_px'],idx[r['i']]['pdf'][-28:],idx[r['i']]['page'],idx[r['j']]['pdf'][-28:],idx[r['j']]['page']))
