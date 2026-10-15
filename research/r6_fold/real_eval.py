"""Real-document evaluation on the private test document (numbers only; images stay in the scratch dir).

Placements of the removal compared (all with the same detector / mask):
  cur  : the shipped pipeline (process_array on the raw page)
  A    : removal on the raw native page, then the unchanged pipeline (edge clean, blank score, clean_page all see the page without the crease)
  C    : the shipped pipeline, then the mask whitened in its output (post-clean)
  A-ls : A with the cleaner's line support disabled (clean_page(line_in=0)) -- how much of the problem is the line support feature
  cur-ls: cur with line support disabled (no removal)
Metrics per fold of the document: crease survival (pixels darker than 10 / 40 gl) in the text-free columns of a +-0.04 in band around the
tracked crease; blank_score of every page; OCR word agreement (tesseract, 200 ppi, single thread) on the pages with text.
Writes research/r6_fold/real_results.json and the scratch images new/ (placement A) and cur/.
"""
import json, os, sys, time
from collections import Counter
import numpy as np, cv2
from common import *
import fold as F
from ocrlib import tess_words
from scanpp.pipeline import process_array
from scanpp.options import Options
from scanpp.clean import clean_page

P = F.FoldParams()
PAIR = {3: 4, 4: 3, 5: 6, 6: 5, 9: 10, 10: 9, 11: 12, 12: 11, 13: 14, 14: 13}
GT = {4: [(.502, .25)], 6: [(.5, 1.0)], 14: [(.336, 0)], 13: [(.339, 0), (.68, -.25)], 9: [(.337, 0), (.678, -.5)], 11: [(.338, -.25), (.68, 0)], 10: [(.335, .25)], 12: [(.336, 0)]}
OPT = Options(duplex='off', notify=False, deskew=False)      # no deskew: the footprints are measured in the unrotated frame


def pipeline(rgb, **kw):
    out, dpi, st = process_array(rgb.copy(), 300.0, OPT, allow_drop=False, resize=False)
    return out, st


def footprints(p):
    a = td_page(p).astype(np.float32)
    g = a.mean(2)
    H, W = g.shape
    out = []
    for y, ang in GT[p]:
        f = F.Fold(y, ang, 0, 0, (0, 0), 0)
        r0, r1, dens, cink, valid, traj = F._fold_geometry(td_page(p), 300.0, f, P)
        yy = np.arange(dens.shape[0])[:, None]
        dist = np.abs(yy - traj[None, :])
        band = dist <= 0.04 * 300
        ink = ((dens >= 0.55) & (dist <= 0.06 * 300)).any(0)
        full = np.zeros((H, W), bool)
        full[r0:r1] = band & ~ink[None, :]
        out.append(full)
    return out, g


def surv(img, fp):
    L = img.astype(np.float32).mean(2)
    return [(int((L[b] < 245).sum()), int((L[b] < 215).sum())) for b in fp]


def words_set(words, min_conf=60.0):
    return Counter(w['text'].strip('.,;:!?()[]"\'') .lower() for w in words if w['conf'] >= min_conf and len(w['text']) > 1)


def f1(a, b):
    inter = sum((a & b).values())
    pa, pb = sum(a.values()), sum(b.values())
    if pa == 0 or pb == 0:
        return 0.0
    p, r = inter / pb, inter / pa
    return 2 * p * r / (p + r) if p + r else 0.0


def main():
    res = {'pages': {}}
    cands = {p: F.line_candidates(td_page(p), 300.0, P) for p in range(1, 15)}
    os.makedirs(SCR + '/cur', exist_ok=True)
    os.makedirs(SCR + '/new', exist_ok=True)
    os.makedirs(SCR + '/newC', exist_ok=True)
    for p in range(1, 15):
        raw = td_page(p)
        cs, (wi, hi) = cands[p]
        part = cands[PAIR[p]][0] if p in PAIR else None
        folds = F.decide(cs, wi, hi, P, part)
        t = time.process_time()
        folds_n = F.detect_folds(raw, 300.0, P, partner=part)            # full API incl. native rule check
        t_detect = time.process_time() - t
        r = dict(folds=[(round(f.y_frac, 3), f.info['tier']) for f in folds_n])
        # placements
        cur, st_cur = pipeline(raw)
        t = time.process_time()
        mask = F.fold_mask(raw, 300.0, folds_n, P)
        rem = F.remove_folds(raw, 300.0, folds_n, P, mask=mask)
        t_remove = time.process_time() - t
        A, st_A = pipeline(rem)
        C = cur.copy()
        C[mask > 0] = 255
        cur_ls = clean_page(raw, 300.0, line_in=0)
        A_ls = clean_page(rem, 300.0, line_in=0)
        np.save(f'{SCR}/cur/p{p:02d}.npy', cur)
        np.save(f'{SCR}/new/p{p:02d}.npy', A)
        np.save(f'{SCR}/newC/p{p:02d}.npy', C)
        r['blank_score'] = dict(cur=st_cur['blank_score'], A=st_A['blank_score'])
        r['t_detect'] = round(t_detect, 3)
        r['t_remove'] = round(t_remove, 3)
        r['mask_px'] = int(mask.sum())
        if p in GT:
            fp, g = footprints(p)
            r['survival_px_lt245_lt215'] = {k: surv(v, fp) for k, v in dict(raw=np.repeat(g[..., None], 3, -1), cur=cur, A=A, C=C, cur_ls=cur_ls, A_ls=A_ls).items()}
        res['pages'][p] = r
        print(p, r['folds'], r['blank_score'], r.get('survival_px_lt245_lt215', {}).get('cur'), r.get('survival_px_lt245_lt215', {}).get('A'), flush=True)
    json.dump(res, open('real_results.json', 'w'), indent=1)


def ocr_part():
    """Word agreement of tesseract (200 ppi, conf >= 60) between the shipped output and the output with removal; plus crease-as-text junk."""
    out = {}
    for p in (13, 9, 11, 3, 5, 1):
        cur = np.load(f'{SCR}/cur/p{p:02d}.npy')
        new = np.load(f'{SCR}/new/p{p:02d}.npy')
        raw = td_page(p)
        wc, tc = tess_words(cur, 300.0, ocr_dpi=200.0)
        wn, tn = tess_words(new, 300.0, ocr_dpi=200.0)
        wr, tr_ = tess_words(raw, 300.0, ocr_dpi=200.0)
        sc, sn, sr = words_set(wc), words_set(wn), words_set(wr)
        out[p] = dict(n_cur=sum(sc.values()), n_new=sum(sn.values()), n_raw=sum(sr.values()),
                      f1_cur_vs_raw=round(f1(sr, sc), 4), f1_new_vs_raw=round(f1(sr, sn), 4), f1_new_vs_cur=round(f1(sc, sn), 4),
                      cpu_tess_s=round(tc, 2),
                      only_cur=sum((sc - sn).values()), only_new=sum((sn - sc).values()),
                      low_conf_words_cur=sum(1 for w in wc if w['conf'] < 60), low_conf_words_new=sum(1 for w in wn if w['conf'] < 60))
        print(p, out[p], flush=True)
    # the same on strips: full-width bands of +-0.6 in around every fold row (psm 6: one block, so the page layout cannot change between variants)
    strips = {}
    for p, ys in {13: [0.338, 0.681], 9: [0.335, 0.68], 11: [0.339, 0.68]}.items():
        for y in ys:
            cur = np.load(f'{SCR}/cur/p{p:02d}.npy'); new = np.load(f'{SCR}/new/p{p:02d}.npy'); raw = td_page(p)
            h = cur.shape[0]
            a, b = int(y * h - 0.6 * 300), int(y * h + 0.6 * 300)
            res_ = {}
            for tag, im in (('cur', cur), ('new', new), ('raw', raw)):
                w_, c_ = tess_words(np.ascontiguousarray(im[a:b]), 300.0, ocr_dpi=200.0, psm=6)
                res_[tag] = words_set(w_)
                res_[tag + '_n_low'] = sum(1 for w in w_ if w['conf'] < 60)
            strips[f'p{p}@{y}'] = dict(n_cur=sum(res_['cur'].values()), n_new=sum(res_['new'].values()), n_raw=sum(res_['raw'].values()), f1_cur_vs_raw=round(f1(res_['raw'], res_['cur']), 4),
                                       f1_new_vs_raw=round(f1(res_['raw'], res_['new']), 4), f1_new_vs_cur=round(f1(res_['cur'], res_['new']), 4), low_conf_cur=res_['cur_n_low'], low_conf_new=res_['new_n_low'])
            print(p, y, strips[f'p{p}@{y}'], flush=True)
    r = json.load(open('real_results.json'))
    r['ocr_strips'] = strips
    r['ocr'] = out
    json.dump(r, open('real_results.json', 'w'), indent=1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'ocr':
        ocr_part()
    else:
        main()
