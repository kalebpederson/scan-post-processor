import json
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
idx = json.load(open(_ROOT + '/research/r4_geometry/index.json'))
def r2_sheets():
    P = json.load(open(_ROOT + '/research/r2_cleaning/pairs_verified.json'))
    key = {(e['pdf'], e['page']): k for k, e in enumerate(idx)}
    S = {}
    for p in P:
        a, b = sorted((p['front'], p['back']))
        S[(p['src'], a, b)] = (key[(p['src'], a - 1)], key[(p['src'], b - 1)])
    return S
if __name__ == '__main__':
    for k, v in r2_sheets().items(): print(k[0][-40:], k[1], k[2], v)
