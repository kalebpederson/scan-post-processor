"""Tesseract as evidence for the fold study (hypothesis of the user).

A. Behaviour on the corpus content classes (single thread, cleaned page, 150 and 200 ppi): words found, confident words (conf >= 60), the fraction of the
   page's ink that lies inside a confident word box (= how much a "protect inside word boxes" rule would cover), CPU per page.
B. OCR as evidence that a candidate line is NOT text: for the stage-1 lines near customary fold rows (score >= 0.4) of the corpus and for the true folds of the
   test document: fraction of the line's length (+-0.02 in) that runs inside a confident (>= 60) / any (>= 0) word box of the page.
(The protection use -- never remove crease pixels inside confident word boxes -- is measured in synth_eval.py with R6_OCR=1: variants vocr60 / vocr30.)
Writes ocr_results.json.
"""
import json, os, sys
from multiprocessing import Pool
import numpy as np, cv2
from common import *
import fold as F
from ocrlib import tess_words
from scanpp.clean import clean_page

P = F.FoldParams()


def page_classes(n_per_class=8, seed=3):
    CL = json.load(open(SCR + '/classify.json'))
    neg = json.load(open('negset.json')) if os.path.exists('negset.json') else {'classes': {}}
    H = {int(k): v for k, v in CL.items()}
    ids = np.arange(len(index()))
    st = np.array([H[i]['strong'] / (H[i]['H'] * H[i]['W']) for i in ids])
    col = np.array([H[i]['col'] / (H[i]['H'] * H[i]['W']) for i in ids])
    p99 = np.array([H[i]['p99'] for i in ids])
    fr = np.array([H[i]['frac'] for i in ids])
    ok = np.array([H[i]['H'] >= 900 and H[i]['W'] >= 700 for i in ids])
    classes = {'printed_text': ok & (st > 0.01) & (col < 0.001), 'handwriting_colour': ok & (col > 0.01), 'pencil_or_faint': ok & (p99 < 0.5) & (fr > 0.006)}
    rng = np.random.default_rng(seed)
    out = []
    for c, m in classes.items():
        for i in rng.choice(ids[m], size=min(n_per_class, int(m.sum())), replace=False):
            out.append((int(i), c))
    return out


def coverage(words, gray_ink, conf, pad):
    bm = np.zeros(gray_ink.shape, np.uint8)
    for w in words:
        if w['conf'] >= conf:
            bm[max(0, int(w['y0']) - pad):int(w['y1']) + pad, max(0, int(w['x0']) - pad):int(w['x1']) + pad] = 1
    return float(bm[gray_ink].mean()) if gray_ink.any() else float('nan'), bm


def classwork(args):
    i, cls = args
    try:
        a, rec = native_page(i)
        dpi = float(rec['dpi'])
        c = clean_page(a, dpi)
        out = dict(idx=i, cls=cls, dpi=dpi)
        for od in (150.0, 200.0):
            od_eff = min(od, dpi)
            words, cpu = tess_words(c, dpi, ocr_dpi=od_eff)
            g = c.mean(2)
            ink = g < 128
            s = od_eff / dpi
            ink_s = cv2.resize(ink.astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_AREA) > 0 if s < 0.999 else ink
            ws = [dict(w, x0=w['x0'] * s, x1=w['x1'] * s, y0=w['y0'] * s, y1=w['y1'] * s) for w in words]
            cov60, _ = coverage(ws, ink_s, 60, 3)
            cov30, _ = coverage(ws, ink_s, 30, 3)
            out[f'ocr{int(od)}'] = dict(n_words=len(words), n_conf60=sum(1 for w in words if w['conf'] >= 60), mean_conf=float(np.mean([w['conf'] for w in words])) if words else 0.0,
                                         ink_in_conf60=cov60, ink_in_conf30=cov30, cpu=cpu, ink_px=int(ink_s.sum()))
        return out
    except Exception as e:
        return dict(idx=i, cls=cls, err=repr(e))


def linework(args):
    k, ys = args
    try:
        a, rec = native_page(k) if k >= 0 else (None, None)
        dpi = float(rec['dpi'])
        words, cpu = tess_words(a, dpi, ocr_dpi=150.0)
        h, w = a.shape[:2]
        out = []
        for y in ys:
            row = int(y * h)
            tol = int(0.02 * dpi)
            cov = {}
            for conf in (60, 0):
                m = np.zeros(w, bool)
                for wd in words:
                    if wd['conf'] >= conf and wd['y0'] - tol <= row <= wd['y1'] + tol:
                        m[max(0, int(wd['x0'])):int(wd['x1'])] = True
                cov[conf] = float(m.mean())
            out.append(dict(y=y, cov60=cov[60], cov0=cov[0]))
        return dict(page=k, lines=out, cpu=cpu)
    except Exception as e:
        return dict(page=k, err=repr(e))


def main():
    res = {}
    with Pool(8) as p:
        R = [r for r in p.map(classwork, page_classes(), chunksize=1)]
    R = [r for r in R if 'err' not in r]
    agg = {}
    for cls in ('printed_text', 'handwriting_colour', 'pencil_or_faint'):
        rows = [r for r in R if r['cls'] == cls]
        d = {}
        for key in ('ocr150', 'ocr200'):
            d[key] = {k: round(float(np.nanmean([r[key][k] for r in rows])), 4) for k in ('n_words', 'n_conf60', 'mean_conf', 'ink_in_conf60', 'ink_in_conf30', 'cpu')}
            d[key]['ink_in_conf60_min'] = round(float(np.nanmin([r[key]['ink_in_conf60'] for r in rows])), 4)
        d['n_pages'] = len(rows)
        agg[cls] = d
    res['class_behaviour'] = agg
    # B: candidate lines
    sw = json.load(open(SCR + '/sweep3.json'))
    jobs = {}
    for k, v in sw.items():
        if 'err' in v or v['wi'] < 5:
            continue
        ys = [c['y_frac'] for c in F._uncrowded(v['c'], P) if c['score'] >= 0.4 and F._near_prior(c['y_frac'], P) and c['amp'] >= P.amp_min]
        if ys:
            jobs[int(k)] = ys
    with Pool(8) as p:
        LW = p.map(linework, list(jobs.items()), chunksize=1)
    covs = [l for r in LW if 'err' not in r for l in r['lines']]
    res['corpus_candidate_lines'] = dict(n_lines=len(covs), n_pages=len(jobs), cov60_mean=float(np.mean([c['cov60'] for c in covs])), cov60_median=float(np.median([c['cov60'] for c in covs])),
                                         cov0_mean=float(np.mean([c['cov0'] for c in covs])), frac_lines_cov60_gt_0_2=float(np.mean([c['cov60'] > 0.2 for c in covs])), cpu_per_page=float(np.mean([r['cpu'] for r in LW if 'err' not in r])))
    # test document folds
    td = {}
    for p_, ys in {13: [.339, .681], 9: [.337, .678], 11: [.338, .68], 4: [.502], 6: [.5], 14: [.336]}.items():
        a = td_page(p_)
        words, cpu = tess_words(a, 300.0, ocr_dpi=150.0)
        h, w = a.shape[:2]
        for y in ys:
            row = int(y * h); tol = int(0.02 * 300)
            m = np.zeros(w, bool)
            for wd in words:
                if wd['conf'] >= 60 and wd['y0'] - tol <= row <= wd['y1'] + tol:
                    m[max(0, int(wd['x0'])):int(wd['x1'])] = True
            td[f'p{p_}@{y}'] = round(float(m.mean()), 3)
    res['test_document_fold_cov60'] = td
    json.dump(res, open('ocr_results.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    import os
    main()
