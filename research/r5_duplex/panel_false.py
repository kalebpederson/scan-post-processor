import sys, os, json, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import duplex
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
R4 = _ROOT + '/research/r4_geometry/'
idx = json.load(open(R4 + 'index.json'))
rd = lambda i: np.ascontiguousarray(cv2.imread(R4 + idx[i]['fn'])[:, :, ::-1])
F = json.load(open('work/fine.json'))
neg = [r for r in F if r['kind'] in ('far', 'cross') and r.get('ncc') is not None and (r['i'], r['j']) not in {(1937, 1939), (521, 404)}]
def srcc(r): return r['dB']['chroma_px'] if r['best'] == 'a' else r['dA']['chroma_px']
dense = sorted([r for r in neg if srcc(r) >= 400], key=lambda r: -r['ncc'])[0]
sparse = [r for r in neg if (r['i'], r['j']) == (1834, 1846)][0]
rows = []
for r, name in ((dense, 'non-pair, dense pages (highest-ncc negative)'), (sparse, 'non-pair, near-blank template pages (highest-gap negative)')):
    A, B = rd(r['i']), rd(r['j'])
    mA, mB = duplex.ink_maps(A, 100.), duplex.ink_maps(B, 100.)
    f = duplex.register_pair(mA, mB); ok = duplex.accept(f, mA, mB) if f.get('reg') is not None else False
    sm = lambda x: cv2.resize(x, (360, 466), interpolation=cv2.INTER_AREA)
    pa = sm(np.ascontiguousarray(A[:, ::-1])); pb = sm(B)
    out = duplex._clean.clean_page(B, 100.); 
    s = duplex.bleed_suppression(A, B, f['reg']) if f.get('reg') is not None else np.zeros(B.shape[:2], np.float32)
    forced = duplex.apply_suppression(out, s)
    panel = np.hstack([pa, pb, sm(out), sm(out if not ok else forced)])
    txt = ['%s: gap %.2f/%.2f  agree %s in  ncc %.2f  colour-source px %d  ->  %s' % (name, r['ga'], r['gb'], None if r.get('agree_in') is None else round(r['agree_in'], 2), r['ncc'], srcc(r), 'ACCEPT' if ok else 'DECLINED: B unchanged'),
           'cols: mirrored A | B raw | B current scanpp | B after duplex pass (identical when declined)']
    hdr = np.full((36, panel.shape[1], 3), 255, np.uint8)
    for k, t in enumerate(txt): cv2.putText(hdr, t, (4, 14 + 15 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 0, 0), 1, cv2.LINE_AA)
    rows.append(np.vstack([hdr, panel])); print(name, ok, 'forced-suppression removed px', int(((out.min(-1) < 250) & (forced.min(-1) >= 250)).sum()), 'if gate bypassed')
cv2.imwrite('panels/false_pair_declined.png', np.vstack(rows)[:, :, ::-1])
