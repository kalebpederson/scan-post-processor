"""Adversarial / missing negatives, drawn onto real blank corpus pages (the corpus has no notebook ruled paper and no bubble sheets).

 ruled_college / ruled_wide : light-blue notebook rules (every 0.28 / 0.34 in, full width, + red margin rule) -- periodic
 hairline_<gray>_<w>        : ONE straight full-width gray rule at row 0.5 (gray level 235 / 215 / 190 / 160, 1 or 3 px at 300 ppi): the hardest look-alike
 dashed_full                : full-width dashed gray line at 0.5
 pencil_short               : a 3-inch graphite line at 0.5 (a 'small pencil line')
 highlighter_band           : full-width yellow highlighter band 0.25 in tall at 1/3
 underline_row              : 2 short underlines (2 in) at 0.5
Reports whether the single-page detector (tier A) fires.  Usage: python adversarial.py -> adversarial_results.json
"""
import json
import numpy as np, cv2
from common import *
import fold as F
from scanpp import geometry

P = F.FoldParams()


def blank_pages(n=3):
    CL = json.load(open(SCR + '/classify.json'))
    ids = [int(k) for k, v in CL.items() if v['frac'] < 0.0015 and v['H'] >= 900 and v['W'] >= 700]
    return ids[:n]


def draw(img, kind, dpi, rng):
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    def hline(y, x0, x1, rgb, wpx, dash=None):
        yy = int(round(y))
        for dy in range(wpx):
            row = yy + dy - wpx // 2
            if dash:
                on = ((np.arange(x0, x1) // dash[0]) % 2 == 0)
                xs = np.arange(x0, x1)[on]
            else:
                xs = np.arange(x0, x1)
            out[row, xs] = np.minimum(out[row, xs], rgb)
    if kind.startswith('ruled'):
        gap = (0.28 if 'college' in kind else 0.34) * dpi
        for y in np.arange(0.9 * dpi, h - 0.4 * dpi, gap):
            hline(y, 0, w, (175, 205, 235), max(1, int(dpi / 300.0 + 0.5)))
        out[:, int(1.2 * dpi):int(1.2 * dpi) + max(1, int(dpi / 300 + 0.5))] = np.minimum(out[:, int(1.2 * dpi):int(1.2 * dpi) + max(1, int(dpi / 300 + 0.5))], (235, 140, 150))
    elif kind.startswith('hairline'):
        g = int(kind.split('_')[1]); wpx = int(kind.split('_')[2])
        hline(0.5 * h, 0, w, (g, g, g + 6), max(1, int(round(wpx * dpi / 300))))
    elif kind == 'dashed_full':
        hline(0.5 * h, 0, w, (190, 190, 196), max(1, int(dpi / 300 + 0.5)), dash=(int(0.08 * dpi), 0))
    elif kind == 'pencil_short':
        x0 = int(rng.uniform(0.2, 0.5) * w)
        hline(0.5 * h, x0, x0 + int(3 * dpi), (150, 150, 152), max(2, int(3 * dpi / 300)))
    elif kind == 'highlighter_band':
        y = int(h / 3)
        out[y:y + int(0.25 * dpi)] = np.minimum(out[y:y + int(0.25 * dpi)], (255, 240, 110))
    elif kind == 'underline_row':
        for x0 in (0.15, 0.55):
            hline(0.5 * h, int(x0 * w), int(x0 * w + 2 * dpi), (60, 60, 60), max(1, int(dpi / 150)))
    out += rng.normal(0, 1.0, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    res = {}
    rng = np.random.default_rng(0)
    kinds = ['ruled_college', 'ruled_wide', 'hairline_235_1', 'hairline_235_3', 'hairline_215_1', 'hairline_215_3', 'hairline_190_1', 'hairline_190_3', 'hairline_160_1', 'hairline_160_3',
             'dashed_full', 'pencil_short', 'highlighter_band', 'underline_row']
    for k in kinds:
        det = []
        for i in blank_pages():
            a, rec = native_page(i)
            dpi = float(rec['dpi'])
            a = geometry._clean_edges_g(np.array(a), dpi, inplace=True)[0]
            a = draw(a, k, dpi, rng)
            fs = F.detect_folds(a, dpi, P)
            cs, (wi, hi) = F.line_candidates(a, dpi, P)
            best = max([c['score'] for c in cs if abs(c['y_frac'] - 0.5) < 0.02], default=0.0)
            det.append(dict(page=i, dpi=dpi, detected=len(fs) > 0, folds=[(round(f.y_frac, 3), f.info['tier'], round(f.info.get('rule_frac', -1), 2)) for f in fs], best_score_near_half=round(best, 2)))
        res[k] = det
        print(k, [(d['detected'], d['best_score_near_half']) for d in det])
    json.dump(res, open('adversarial_results.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
