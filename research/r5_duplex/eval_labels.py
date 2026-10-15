"""R2 faint-content retention (17 labelled regions) before/after the duplex post-pass, using each labelled page's neighbours as sheet partner
(forced registration = worst case, bypassing the gate; and the gate decision)."""
import sys, os, json, hashlib, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
R2 = _ROOT + '/research/r2_cleaning'; ROOT = _ROOT
import duplex
sys.path.insert(0, R2); cwd = os.getcwd(); os.chdir(R2)
from common import load_rgb, darkness
from evaluate import dE_white
os.chdir(cwd)
LAB = json.load(open(R2 + '/labels.json')); man = json.load(open(ROOT + '/research/corpus/manifest.json'))
slug = lambda src: hashlib.md5(src.encode()).hexdigest()[:8]
def page_of(f):
    n = os.path.basename(f)
    for e in man:
        if e['file'] == n: return (e['source'], e['page'] - 1) if e['page'] else (None, None)
    if 'work/pages/' in f:
        sl, pg = f.split('work/pages/')[1].split('/'); return sl, int(pg[3:6])
    return None, None
def partner_file(f, delta):
    src, p = page_of(f)
    if src is None: return None
    sl = slug(src) if src.endswith('.pdf') else src
    q = f'{R2}/work/pages/{sl}/pg-{p + delta:03d}.jpg'
    return q if os.path.exists(q) else None
def main():
    rows = []
    byfile = {}
    for r in LAB['pos']: byfile.setdefault(r['file'], []).append(r)
    for f, regs in byfile.items():
        path = ROOT + '/' + f
        if 'epson' in f: continue
        rgb = load_rgb(path); dpi = regs[0]['dpi']; h, w = rgb.shape[:2]
        d, _ = darkness(rgb, dpi); ds = cv2.GaussianBlur(d, (0, 0), max(0.6, 0.006 * dpi))
        out = duplex._clean.clean_page(rgb, dpi)
        mM = duplex.ink_maps(rgb, dpi)
        for delta in (-1, 1):
            pf = partner_file(f, delta)
            if pf is None: continue
            src = load_rgb(pf)
            if src.shape != rgb.shape: continue
            mS = duplex.ink_maps(src, dpi)
            ft = duplex.register_pair(mS, mM, coarse_gap_min=0.0)
            if ft.get('reg') is None: continue
            s = duplex.bleed_suppression(src, rgb, ft['reg']); new = duplex.apply_suppression(out, s)
            for r in regs:
                x, y, bw, bh = r['box']; m = np.zeros((h, w), bool); m[y:y+bh, x:x+bw] = True
                v = ds[m & (ds >= 0.03)]
                if len(v) < 20: continue
                thr = max(0.04, 0.5 * float(np.percentile(v, 95))); S = m & (ds >= thr) & (ds <= r['dmax'])
                if S.sum() < 20: continue
                ret = lambda o: float((dE_white(o[S]) >= 6).mean())
                rows.append(dict(id=r['id'], cls=r['cls'], delta=delta, ret_cur=ret(out), ret_new=ret(new), n=int(S.sum()), s_in_region=float((s[S] > .5).mean()),
                                 gate_ncc=ft.get('ncc'), gmax=max(ft['ga'], ft['gb']), gate=bool(duplex.accept(ft, mS, mM))))
    json.dump(rows, open('labels_eval.json', 'w'), indent=1)
    for r in rows: print(r['id'], r['delta'], 'cur %.3f forced-new %.3f  suppressed-in-core %.3f  gmax %.2f gate %s' % (r['ret_cur'], r['ret_new'], r['s_in_region'], r['gmax'], r['gate']))
    print('mean ret cur %.4f forced %.4f gated %.4f over %d region/partner pairs; gate accepted %d' % (np.mean([r['ret_cur'] for r in rows]), np.mean([r['ret_new'] for r in rows]), np.mean([r['ret_new'] if r['gate'] else r['ret_cur'] for r in rows]), len(rows), sum(r['gate'] for r in rows)))
main()
