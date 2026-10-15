"""Panels for research/r6_fold/panels/ (corpus pages and synthetic creases only -- never the private test document).

  neg_<class>.png     : negative pages. raw | shipped cleaner | cleaner + fold pipeline (the detector does not fire: identical, 0 px changed) | STRESS: the removal
                        forced onto the strongest horizontal line of the page, to show what a false detection would cost.
  synth_<cls>.png     : synthetic crease (real strip from the fold-only backs p4 / p6 / p14 / p13) pasted over a corpus page.
                        reference (no crease) | pasted raw | shipped cleaner | cleaner + fold removal (placement A)
Usage: python panels.py [neg|synth]   (instances are chosen from negset.json / synth_instances.json)
"""
import json, sys
import numpy as np, cv2
from common import *
import fold as F
from scanpp import geometry
from scanpp.clean import clean_page
from synth_eval import load_strip, load_wander, paste

P = F.FoldParams()
OUT = ROOT + '/research/r6_fold/panels/'


def label(img, txt):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (8 + 7 * len(txt), 16), (255, 255, 255), -1)
    cv2.putText(img, txt, (3, 12), 0, 0.42, (0, 0, 200), 1)
    return img


def crop_rows(img, y, half, x0, x1):
    return img[max(0, y - half):y + half, x0:x1]


def neg_panel(idx, name, x0=0, w=1000):
    rgb, rec = native_page(idx)
    dpi = float(rec['dpi'])
    rgb = geometry._clean_edges_g(np.array(rgb), dpi, inplace=True)[0]
    h = rgb.shape[0]
    cs, (wi, hi) = F.line_candidates(rgb, dpi, P)
    folds = F.decide(cs, wi, hi, P, None)
    cur = clean_page(rgb, dpi)
    out = clean_page(F.remove_folds(rgb, dpi, folds, P), dpi) if folds else cur
    changed = int((out != cur).any(2).sum())
    # stress: force removal on the highest-scoring candidate (any score) of the page, else the best near-prior row
    cs2 = sorted(cs, key=lambda c: -c['score'])
    y = cs2[0]['y_frac'] if cs2 else 0.5
    ang = cs2[0]['angle'] if cs2 else 0.0
    forced = [F.Fold(y, ang, 0, 1, (1, 1), 99)]
    stress = clean_page(F.remove_folds(rgb, dpi, forced, P), dpi)
    yc, half = int(y * h), int(0.45 * dpi)
    x1 = min(rgb.shape[1], x0 + int(w * dpi / 200))
    tiles = [label(crop_rows(im, yc, half, x0, x1), t) for im, t in ((rgb, 'raw'), (cur, 'shipped cleaner'), (out, 'with fold pipeline: %d px changed (%d folds)' % (changed, len(folds))),
                                                                  (stress, 'STRESS: forced removal at y=%.3f' % y))]
    hh = min(t.shape[0] for t in tiles)
    img = np.vstack([np.vstack([t[:hh], np.full((3, t.shape[1], 3), 128, np.uint8)]) for t in tiles])
    cv2.imwrite(OUT + name, img[:, :, ::-1])
    return dict(idx=idx, page=rec['pdf'].split('/')[-1][:30], folds=len(folds), changed_px=changed, stress_row=round(y, 3))


def synth_panel(idx, strip, y, tilt, name, x0=0, w=1000):
    page, rec = native_page(idx)
    dpi = float(rec['dpi'])
    page = geometry._clean_edges_g(np.array(page), dpi, inplace=True)[0]
    h = page.shape[0]
    pasted, T = paste(page, load_strip(strip, dpi), y * h, tilt, dpi, load_wander(strip))
    folds = F.detect_folds(pasted, dpi, P)
    ref = clean_page(page, dpi)
    cur = clean_page(pasted, dpi)
    new = clean_page(F.remove_folds(pasted, dpi, folds, P), dpi) if folds else cur
    yc, half = int(y * h), int(0.35 * dpi)
    x1 = min(page.shape[1], x0 + int(w * dpi / 200))
    tiles = [label(crop_rows(im, yc, half, x0, x1), t) for im, t in ((ref, 'reference: page without crease, shipped cleaner'), (pasted, 'crease (real strip %s) pasted, raw' % strip), (cur, 'shipped cleaner'),
                                                                   (new, 'cleaner + fold removal: detected %s' % [(round(float(f.y_frac), 3), f.info['tier']) for f in folds]))]
    hh = min(t.shape[0] for t in tiles)
    img = np.vstack([np.vstack([t[:hh], np.full((3, t.shape[1], 3), 128, np.uint8)]) for t in tiles])
    cv2.imwrite(OUT + name, img[:, :, ::-1])


def adv_panel():
    import adversarial as A
    rng = np.random.default_rng(0)
    page_id = A.blank_pages(1)[0]
    base, rec = native_page(page_id)
    dpi = float(rec['dpi'])
    base = geometry._clean_edges_g(np.array(base), dpi, inplace=True)[0]
    rows = []
    info = {}
    for kind in ('ruled_college', 'hairline_215_1', 'hairline_190_3', 'dashed_full', 'pencil_short', 'highlighter_band'):
        a = A.draw(base, kind, dpi, rng)
        folds = F.detect_folds(a, dpi, P)
        cur = clean_page(a, dpi)
        new = clean_page(F.remove_folds(a, dpi, folds, P), dpi) if folds else cur
        h, w = a.shape[:2]
        yc = int((1 / 3 if kind == 'highlighter_band' else 0.5) * h)
        half = int(0.5 * dpi)
        x0, x1 = 0, min(w, int(900 * dpi / 200))
        t = [label(crop_rows(im, yc, half, x0, x1), tx) for im, tx in ((a, kind + ': synthetic on a blank corpus page'), (cur, 'shipped cleaner'), (new, 'with fold pipeline: %d fold(s) detected, %d px changed' % (len(folds), int((cur != new).any(2).sum()))))]
        hh = min(x.shape[0] for x in t)
        rows.append(np.vstack([x[:hh] for x in t] + [np.full((4, t[0].shape[1], 3), 128, np.uint8)]))
        info[kind] = dict(detected=len(folds), changed_px=int((cur != new).any(2).sum()))
    cv2.imwrite(OUT + 'adversarial.png', np.vstack(rows)[:, :, ::-1])
    json.dump(info, open('panels_adv.json', 'w'), indent=1)
    print(info)


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'neg'
    if which == 'adv':
        adv_panel()
    if which == 'synth':
        R = json.load(open('synth_instances.json'))
        pick = {}
        for r in R:
            if r['mode'] == 'e2e' and r['det'] and (r['ret_ref'] - r['ret_vdefault']) < 0.02 and r['surv40_vdefault'] < 0.5 * max(1, r['foot40']):
                pick.setdefault(r['cls'], []).append(r)
        for cls, rr in pick.items():
            r = sorted(rr, key=lambda r: -r['ncore'])[0]
            synth_panel(r['idx'], r['strip'], r['y'], r['tilt'], f'synth_{cls}.png')
            print(cls, r['idx'], r['strip'], r['y'])
    if which == 'neg':
        neg = json.load(open('negset.json'))['classes']
        dr = json.load(open('detect_results.json'))
        info = {}
        for cls in ('ruled', 'graph', 'table', 'rule', 'dashed', 'highlighter', 'pencil', 'handwriting', 'hstroke'):
            ids = neg.get(cls, [])
            if not ids:
                continue
            nc = dr['negative_classes'][cls]
            # the closest negative of the class (highest near-prior score) that is not a detection
            pick = [i for s_, i in nc['top_near_prior_scores'] if i not in nc['tierA_pages'] and i not in (942, 943, 1042, 1043)]      # (the pink creased pages are positives)
            info[cls] = neg_panel(pick[0] if pick else ids[len(ids) // 2], f'neg_{cls}.png')
        json.dump(info, open('panels_neg.json', 'w'), indent=1)
        print(info)
