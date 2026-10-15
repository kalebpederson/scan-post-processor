"""Negative set: look-alike structures of fold creases, found in the corpus (R4 100-ppi cache + the loose scan images) by simple structural proxies.

Classes (proxy definitions, all at 100 ppi on the ink density of fold.density_maps; lengths in inches):
  ruled      >= 8 long (>= 3 in) horizontal line segments with regular spacing (CV <= 0.2) and <= 2 long vertical ones
  graph      >= 10 horizontal AND >= 10 vertical segments of >= 0.8 in (grid)
  table      >= 3 long (>= 3 in) horizontal and >= 3 long (>= 1 in) vertical segments, not graph
  rule       1-2 isolated long (>= 3 in) horizontal segments and no table (underlines, form / answer lines, header rules)
  dashed     >= 1 row with >= 8 short thin horizontal components at a regular gap, spanning >= 1.5 in
  highlighter  >= 0.15 % of the page is saturated colour (HSV S > 0.5, V > 0.7) in horizontal bands >= 0.8 in long
  pencil     page-level ink density p99.5 < 0.5 with ink (graphite / light pen, R2 'pencil' pages) -- plus the R2-labelled pencil pages
  handwriting  >= 1 % of the page is coloured pen ink (chroma ratio >= 0.4, density >= 0.25)  (teacher / student pens)
  hstroke    >= 2 hand-drawn horizontal pen strokes (coloured-ink component, aspect >= 10, length 0.5-3 in, <= 0.06 in tall): underlines, fraction bars, crossings-out
Output: research/r6_fold/negset.json (class -> page ids, counts) ; the detector's decision on each member is evaluated in detect_eval.
Usage: python negset.py   (14 processes, ~1 min)
"""
import json
from multiprocessing import Pool
import numpy as np, cv2
from common import *
import fold as F

P = F.FoldParams()


def segs(mask, kern, vertical=False):
    k = np.ones((kern, 1) if vertical else (1, kern), np.uint8)
    o = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(o, connectivity=8)
    return st[1:] if n > 1 else np.zeros((0, 5), int)


def features(rgb, dpi=100.0):
    dens, cink, d = F.density_maps(rgb, dpi, P, native=True)
    h, w = dens.shape
    ink = (dens > 0.2).astype(np.uint8)
    f = dict(H=h, W=w)
    hs = segs(ink, int(3 * d))
    vs = segs(ink, int(1 * d), True)
    hs_long = hs[hs[:, 3] <= 6] if len(hs) else hs                     # thin
    ys = np.sort((hs_long[:, 1] + hs_long[:, 3] / 2.0)) if len(hs_long) else np.array([])
    # merge segments of the same row
    rows = []
    for y in ys:
        if not rows or y - rows[-1] > 3:
            rows.append(y)
    gaps = np.diff(rows) if len(rows) > 1 else np.array([])
    f['n_h_long'] = len(rows)
    f['n_v_long'] = int(len(vs))
    f['gap_cv'] = float(gaps.std() / gaps.mean()) if len(gaps) > 2 and gaps.mean() > 0 else 9.0
    hs_s = segs(ink, int(0.8 * d))
    vs_s = segs(ink, int(0.8 * d), True)
    f['n_h_short'] = int(len(hs_s))
    f['n_v_short'] = int(len(vs_s))
    f['h2'] = int(len(segs(ink, int(2 * d))))
    f['v1'] = int(len(vs))
    # dashed: rows with many short thin components
    n, lab, st, _ = cv2.connectedComponentsWithStats((dens > 0.15).astype(np.uint8), connectivity=8)
    comp = st[1:]
    thin = comp[(comp[:, 3] <= 4) & (comp[:, 2] >= 3) & (comp[:, 2] <= int(0.3 * d))] if len(comp) else comp
    dashed = 0
    if len(thin):
        yc = thin[:, 1] + thin[:, 3] // 2
        for y in np.unique(yc):
            sel = thin[np.abs(yc - y) <= 1]
            if len(sel) >= 8:
                xs = np.sort(sel[:, 0])
                gp = np.diff(xs)
                if (xs[-1] - xs[0]) >= 1.5 * d and gp.std() / max(gp.mean(), 1) < 0.35:
                    dashed += 1
    f['dashed_rows'] = dashed
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = ((hsv[..., 1] > 128) & (hsv[..., 2] > 178) & (dens > 0.1)).astype(np.uint8)
    f['sat_frac'] = float(sat.mean())
    f['hl_bands'] = int(len(segs(sat, int(0.8 * d)) ))
    f['p995'] = float(np.percentile(dens, 99.5))
    f['ink_frac'] = float((dens > 0.15).mean())
    f['pen_frac'] = float((cink & (dens >= 0.25)).mean())
    # hand-drawn horizontal strokes: coloured (or light) components, aspect >= 8, length 0.5-3 in, not on a long rule
    n, lab, st, _ = cv2.connectedComponentsWithStats((cink & (dens >= 0.25)).astype(np.uint8), connectivity=8)
    c = st[1:]
    ok = (c[:, 2] >= 0.5 * d) & (c[:, 2] <= 3 * d) & (c[:, 2] >= 10 * np.maximum(c[:, 3], 1)) & (c[:, 3] >= 2) & (c[:, 3] <= 0.06 * d) if len(c) else np.zeros(0, bool)
    f['hstrokes'] = int(ok.sum())
    return f


def work(i):
    try:
        rgb, r = corpus_page(i)
        return i, features(rgb)
    except Exception as e:
        return i, dict(err=repr(e))


def classify(f, r2_pencil):
    cl = []
    if f['H'] < 700 or f['W'] < 500:
        return ['small']
    if f['n_h_long'] >= 8 and f['gap_cv'] <= 0.2 and f['n_v_long'] <= 2:
        cl.append('ruled')
    if f['n_h_short'] >= 10 and f['n_v_short'] >= 10:
        cl.append('graph')
    if f['n_h_long'] >= 3 and f['n_v_long'] >= 3 and 'graph' not in cl and 'ruled' not in cl:
        cl.append('table')
    if 1 <= f['n_h_long'] <= 2 and f['n_v_long'] < 2:
        cl.append('rule')
    if f['dashed_rows'] >= 1:
        cl.append('dashed')
    if f['hl_bands'] >= 1 and f['sat_frac'] >= 0.0015:
        cl.append('highlighter')
    if (f['p995'] < 0.5 and f['ink_frac'] > 0.003) or r2_pencil:
        cl.append('pencil')
    if f['pen_frac'] >= 0.01:
        cl.append('handwriting')
    if f['hstrokes'] >= 2:
        cl.append('hstroke')
    return cl


if __name__ == '__main__':
    with Pool(14) as p:
        res = dict(p.map(work, range(len(index())), chunksize=8))
    r2_pencil = {445, 86, 201}        # pages of the R2-labelled pencil / grid corpus images that are in the index (see research/r2_cleaning/labels.json)
    out = {'proxy_definitions': __doc__, 'classes': {}, 'per_page': {}}
    for k, f in res.items():
        if 'err' in f:
            continue
        cl = classify(f, int(k) in r2_pencil)
        out['per_page'][k] = cl
        for c in cl:
            out['classes'].setdefault(c, []).append(int(k))
    out['counts'] = {c: len(v) for c, v in out['classes'].items()}
    json.dump(out, open('negset.json', 'w'))
    json.dump(res, open(SCR + '/negset_features.json', 'w'))
    print(out['counts'])
