"""Duplex corroboration of fold creases (hypothesis: a crease runs through the whole sheet, so it shows on both sides at the same row).

1. Test document, the five physical sheets (3,4) (5,6) (9,10) (11,12) (13,14): row of the crease on each side (stage-1 line of the best score near
   the ground-truth rows), their difference, and the sign of the cross-section asymmetry (valley / ridge) on the two sides.
2. Geometric pairing for black-print pages: NCC of the front's ink layer against the mirrored back's faint ink layer (bleed-through ghost) at 100 ppi,
   peak z-score against the null of all other shifts, for the true sheets and for the non-sheet neighbours (2,3) (4,5) (6,7) (8,9) (10,11) (12,13).
3. Corpus: false corroboration rate of the scheme tier B (own >= pair_score_min, partner >= pair_partner_min at the same row +- pair_tol) over all adjacent
   same-PDF page pairs (from scratch/sweep3.json).
Writes duplex_results.json.
"""
import json
import numpy as np, cv2
from common import *
import fold as F

P = F.FoldParams()
SHEETS = [(3, 4), (5, 6), (9, 10), (11, 12), (13, 14)]
NON = [(2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13)]
FOLDS = {3: [.5], 4: [.502], 5: [.5], 6: [.5], 9: [.337, .678], 10: [.335, .679], 11: [.338, .68], 12: [.336, .68], 13: [.339, .681], 14: [.336, .679]}


def profile_asym(p, y, ang):
    """Mean vertical cross-section (gray, +-0.25 in) around the tracked crease in the columns without ink; returns (profile, asymmetry = mean(above) - mean(below))."""
    raw = td_page(p)
    f = F.Fold(y, ang, 0, 0, (0, 0), 0)
    r0, r1, dens, cink, valid, traj = F._fold_geometry(raw, 300.0, f, P)
    g = raw[r0:r1].astype(np.float32).mean(2)
    J = 75
    cols = np.flatnonzero((valid[np.clip(np.round(traj).astype(int)[None, :] + np.arange(-J, J + 1)[:, None], 0, g.shape[0] - 1), np.arange(g.shape[1])[None, :]] > 0).all(0))
    prof = np.zeros(2 * J + 1)
    for x in cols[::4]:
        prof += g[np.clip(int(round(traj[x])) + np.arange(-J, J + 1), 0, g.shape[0] - 1), x]
    prof /= max(1, len(cols[::4]))
    return prof, float(255 - prof[:J - 5].mean() - (255 - prof[J + 6:].mean()))


def ncc_peak(front, back):
    def lay(p, thr):
        g = td_page(p, 100).astype(np.float32).mean(2)
        d = np.clip(1 - g / max(np.percentile(g, 95), 1), 0, 1)
        m = 45                                          # 0.45 in: scanner edge shadows / coil are not content
        d[:m] = 0; d[-m:] = 0; d[:, :m] = 0; d[:, -m:] = 0
        return d
    a = lay(front, 0)
    b = lay(back, 0)[:, ::-1]                              # mirrored back
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    a = cv2.GaussianBlur(np.minimum(a[:h, :w], 0.6), (0, 0), 1.5); b = cv2.GaussianBlur(np.minimum(b[:h, :w], 0.6), (0, 0), 1.5)
    a -= a.mean(); b -= b.mean()
    A = np.fft.rfft2(a); Bf = np.fft.rfft2(b)
    c = np.fft.irfft2(A * np.conj(Bf), s=a.shape)
    c = np.fft.fftshift(c) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
    cy, cx = np.array(c.shape) // 2
    win = c[cy - 40:cy + 41, cx - 40:cx + 41]                # +-0.4 in search window
    pk = float(win.max()); null = c[::3, ::3]
    z = (pk - null.mean()) / (null.std() + 1e-9)
    return pk, float(z), tuple(int(v) for v in (np.unravel_index(win.argmax(), win.shape) - np.array([40, 40])))


def main():
    out = {'sheets': [], 'pairing': {'sheets': [], 'non_sheets': []}}
    # plain ridge candidates (cand.py: Radon ridge amplitude, no decision criteria): the crease row on each side is the strongest ridge within 0.02 of the ground truth
    import cand as C
    cand = {}
    for p in range(1, 15):
        cs, _ = C.candidates(td_page(p, 100), 100.0, n=80)
        cand[p] = [dict(y_frac=c['yf'], angle=c['ang'], amp=c['amp'], score=0.0) for c in cs]
    for fr, bk in SHEETS:
        for y in FOLDS[fr]:
            row = dict(front=fr, back=bk, y_gt=y)
            for side, p in (('front', fr), ('back', bk)):
                m = sorted([c for c in cand[p] if abs(c['y_frac'] - y) <= 0.02], key=lambda c: -c['amp'])
                row[side + '_row'] = round(m[0]['y_frac'], 4) if m else None
                row[side + '_angle'] = m[0]['angle'] if m else None
                row[side + '_amp'] = round(m[0]['amp'], 4) if m else None
                if m:
                    prof, asym = profile_asym(p, m[0]['y_frac'], m[0]['angle'])
                    row[side + '_asym_above_minus_below'] = round(asym, 2)
            if row['front_row'] is not None and row['back_row'] is not None:
                row['dy'] = round(row['front_row'] - row['back_row'], 4)
                row['dy_px300'] = round(row['dy'] * 3270, 1)
            out['sheets'].append(row)
    for tag, prs in (('sheets', SHEETS), ('non_sheets', NON)):
        for a, b in prs:
            pk, z, sh = ncc_peak(a, b)
            out['pairing'][tag].append(dict(pair=[a, b], ncc_peak=round(pk, 4), z=round(z, 1), shift_px100=sh))
    d = json.load(open(SCR + '/sweep3.json')); idx = index()
    def cands(k):
        v = d[str(k)]
        return F._uncrowded(v['c'], P) if v['wi'] >= 5 else []
    grid = {}
    for a_, b_ in [(.5, .5), (.6, .8), (.6, 1.0), (.7, .7), (.8, .8)]:
        n = 0; pages = set(); npairs = 0
        for k in range(len(idx)):
            for j in (k - 1, k + 1):
                if 0 <= j < len(idx) and idx[j]['pdf'] == idx[k]['pdf']:
                    npairs += 1
                    for c in cands(k):
                        if c['score'] >= a_ and any(abs(q['y_frac'] - c['y_frac']) <= P.pair_tol and q['score'] >= b_ for q in cands(j)):
                            n += 1; pages.add(k)
        grid[f'own>={a_}_partner>={b_}'] = dict(corroborated_lines=n, pages=len(pages), adjacent_ordered_pairs=npairs)
    out['corpus_false_corroboration'] = grid
    json.dump(out, open('duplex_results.json', 'w'), indent=1)
    for r in out['sheets']: print(r)
    print(out['pairing']); print(grid)

if __name__ == '__main__':
    main()
