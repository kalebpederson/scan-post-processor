"""Detector evaluation: (1) the whole corpus (R4 100-ppi cache, 1969 pages, edge-cleaned like the pipeline), (2) the 14 pages of the test document.

Stage 1 results of the corpus come from sweep3.py (scratch/sweep3.json; `python sweep3.py`).  Ablations re-run stage 1 with a modified
FoldParams (`python detect_eval.py ablate`, ~90 s each on 14 processes) and are stored in scratch/ablate_<name>.json.

Output: research/r6_fold/detect_results.json (aggregates, page ids, scores; no images, no text).
"""
import json, os, sys
import numpy as np
from common import *
import fold as F

P = F.FoldParams()
# ground truth of the test document, by eye (page, row fraction): folds that survive the current cleaner are "relevant"
TD_FOLDS = {'p3': [0.5], 'p4': [0.502], 'p5': [0.5], 'p6': [0.5], 'p9': [0.337, 0.678], 'p10': [0.335, 0.679], 'p11': [0.338, 0.68], 'p12': [0.336, 0.68],
            'p13': [0.339, 0.681], 'p14': [0.336, 0.679]}
TD_RELEVANT = {'p4': [0.502], 'p6': [0.5], 'p9': [0.337, 0.678], 'p11': [0.338, 0.68], 'p13': [0.339, 0.681], 'p14': [0.336]}
TD_PAIR = {3: 4, 4: 3, 5: 6, 6: 5, 9: 10, 10: 9, 11: 12, 12: 11, 13: 14, 14: 13}


def corpus_decisions(sweep, Pp=P, use_partner=True):
    idx = index()
    A, B = {}, {}
    for k, v in sweep.items():
        k = int(k)
        if 'err' in v:
            continue
        fs = F.decide(v['c'], v['wi'], v['hi'], Pp, None)
        if fs:
            A[k] = [(round(f.y_frac, 3), round(f.amp, 3), round(f.info['score'], 2)) for f in fs]
        if use_partner:
            for j in (k - 1, k + 1):
                if 0 <= j < len(idx) and idx[j]['pdf'] == idx[k]['pdf'] and str(j) in sweep and 'err' not in sweep[str(j)]:
                    fb = [f for f in F.decide(v['c'], v['wi'], v['hi'], Pp, sweep[str(j)]['c']) if f.info['tier'] == 'B']
                    if fb:
                        B[(k, j)] = [(round(f.y_frac, 3), round(f.info['score'], 2)) for f in fb]
    return A, B


def td_stage1():
    out = {}
    for p in range(1, 15):
        cs, (wi, hi) = F.line_candidates(td_page(p), 300.0, P)
        out[p] = (cs, wi, hi)
    return out


def main():
    sweep = json.load(open(SCR + '/sweep3.json'))
    res = {'params': {k: (list(v) if isinstance(v, tuple) else v) for k, v in P.__dict__.items()}}
    idx = index()
    n_pages = sum(1 for v in sweep.values() if 'err' not in v and v['wi'] >= 5 and v['hi'] >= 7)
    res['corpus'] = dict(pages=len(sweep), pages_evaluated=n_pages, small_pages_skipped=len(sweep) - n_pages)
    # decision: tier A / B
    A, B = corpus_decisions(sweep)
    res['corpus']['tierA_pages'] = {str(k): v for k, v in A.items()}
    res['corpus']['tierB_pairs'] = {f'{k}-{j}': v for (k, j), v in B.items()}
    # native-resolution check (dark isolated rule test) of every page the stage-1 decision flagged, on the real page; visual review of every one is in REPORT.md
    from scanpp import geometry
    fin = {}
    for k in sorted(set(A) | {a for a, _ in B}):
        a, rec = native_page(k)
        dpi = float(rec['dpi'])
        a = geometry._clean_edges_g(np.array(a), dpi, inplace=True)[0]
        fs = F.detect_folds(a, dpi, P)
        fin[str(k)] = dict(pdf=rec['pdf'].split('/')[-1][:40], page=rec['page'], dpi=dpi, stage1=[(round(f.y_frac, 3), f.info['tier']) for f in F.decide(*F.line_candidates(a, dpi, P)[0:1], *F.line_candidates(a, dpi, P)[1], P, None)],
                           final=[(round(f.y_frac, 3), f.info['tier'], round(f.info.get('rule_frac', -1), 2), round(f.amp, 3)) for f in fs])
    res['corpus']['native_check'] = fin
    # margins: the best score among near-prior lines that were NOT folds (all of them are negatives or unknown pink folds), and among all lines
    best_prior, best_any, near = [], [], []
    for k, v in sweep.items():
        if 'err' in v or v['wi'] < 5:
            continue
        for c in F._uncrowded(v['c'], P):
            if F._near_prior(c['y_frac'], P):
                best_prior.append((c['score'], int(k), round(c['y_frac'], 3), round(c['amp'], 3)))
            best_any.append((c['score'], int(k), round(c['y_frac'], 3), round(c['amp'], 3)))
    best_prior.sort(reverse=True)
    best_any.sort(reverse=True)
    res['corpus']['top_scores_near_prior'] = best_prior[:15]
    res['corpus']['top_scores_any_row'] = best_any[:15]
    res['corpus']['lines_score_ge_1_any_row'] = sum(1 for s, *_ in best_any if s >= 1.0)
    res['corpus']['lines_score_ge_1_near_prior'] = sum(1 for s, *_ in best_prior if s >= 1.0)
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        res['corpus'][f'lines_near_prior_score_ge_{t}'] = sum(1 for s, *_ in best_prior if s >= t)
    # --- per negative class (negset.json): pages, pages with a tier-A fold, best near-prior score of the class
    if os.path.exists('negset.json'):
        nc = {}
        ns = json.load(open('negset.json'))['classes']
        score_by_page = {}
        for k, v in sweep.items():
            if 'err' in v or v['wi'] < 5:
                continue
            ss = [c['score'] for c in F._uncrowded(v['c'], P) if F._near_prior(c['y_frac'], P)]
            score_by_page[int(k)] = max(ss, default=0.0)
        for cls, ids in ns.items():
            sc = sorted(((score_by_page.get(i, 0.0), i) for i in ids), reverse=True)
            nc[cls] = dict(pages=len(ids), tierA_pages=[i for i in ids if i in A], top_near_prior_scores=[(round(s_, 2), i) for s_, i in sc[:3]])
        res['negative_classes'] = nc
    # --- test document
    td = td_stage1()
    rows = []
    for p in range(1, 15):
        cs, wi, hi = td[p]
        part = td[TD_PAIR[p]][0] if p in TD_PAIR else None
        fs = F.decide(cs, wi, hi, P, part)
        fs_alone = F.decide(cs, wi, hi, P, None)
        gt = TD_FOLDS.get(f'p{p}', [])
        rel = TD_RELEVANT.get(f'p{p}', [])
        for y in gt:
            m = [f for f in fs if abs(f.y_frac - y) <= 0.015]
            ma = [f for f in fs_alone if abs(f.y_frac - y) <= 0.015]
            near = [c for c in cs if abs(c['y_frac'] - y) <= 0.015]
            rows.append(dict(page=p, y=y, relevant=y in rel, detected=bool(m), tier=m[0].info['tier'] if m else None, detected_alone=bool(ma),
                             best_score=round(max([c['score'] for c in near], default=0.0), 2), amp=round(max([c['amp'] for c in near], default=0.0), 4)))
        fp = [f for f in fs if not any(abs(f.y_frac - y) <= 0.015 for y in gt)]
        if fp:
            rows.append(dict(page=p, false_detections=[round(f.y_frac, 3) for f in fp]))
    res['test_document'] = rows
    json.dump(res, open('detect_results.json', 'w'), indent=1)
    print('corpus: pages evaluated', n_pages, '| tier A pages', len(A), '| tier B pairs', len(B))
    print('top scores near prior', [(round(s, 2), k, y) for s, k, y, a in best_prior[:6]])
    print('lines with score>=1: any row', res['corpus']['lines_score_ge_1_any_row'], ' near prior', res['corpus']['lines_score_ge_1_near_prior'])
    for r in rows:
        print(r)


_OFF = dict(pres_frac_min=1e-9, cv_mean_min=1e-9, cont_min=1e-9, obs_frac_min=0.0, amp_min=0.002, amp_max=9.0, chroma_d=9.0, use_edge=False, step_ratio=0.0)
ABLATIONS = {                       # name -> FoldParams overrides (stage 1 re-run)
    # progression: ridge only -> + edge-to-edge -> + presence along the line -> + non-ink along the line -> + amplitude window -> + continuity (-> final = + chroma, crowd)
    'prog0_ridge': dict(_OFF),
    'prog1_edge': dict(_OFF, use_edge=True),
    'prog2_pres': dict(_OFF, use_edge=True, pres_frac_min=0.75),
    'prog3_cvobs': dict(_OFF, use_edge=True, pres_frac_min=0.75, cv_mean_min=0.75, obs_frac_min=0.4),
    'prog4_amp': dict(_OFF, use_edge=True, pres_frac_min=0.75, cv_mean_min=0.75, obs_frac_min=0.4, amp_min=0.02, amp_max=0.14, step_ratio=1.0),
    'prog5_cont': dict(_OFF, use_edge=True, pres_frac_min=0.75, cv_mean_min=0.75, obs_frac_min=0.4, amp_min=0.02, amp_max=0.14, step_ratio=1.0, cont_min=0.55),
    'cov_base_0.7': dict(cov_base=0.7),
    'loose': dict(obs_frac_min=0.3, cov_base=0.7, pres_frac_min=0.65, cont_min=0.45, amp_min=0.015),
    'looser': dict(obs_frac_min=0.2, cov_base=0.6, pres_frac_min=0.6, cont_min=0.4, amp_min=0.012),
    'no_cont': dict(cont_min=1e-9),
    'no_amp_floor': dict(amp_min=0.002),
    'no_obs_frac': dict(obs_frac_min=0.0),
    'no_chroma': dict(chroma_d=9.0),
    'no_edge': dict(use_edge=False),
    'no_crowd': dict(crowd_in=0.0),
}


def ablate(names):
    from multiprocessing import Pool
    summary = json.load(open('ablation_results.json')) if os.path.exists('ablation_results.json') else {}
    for name in names:
        Pa = F.FoldParams(**ABLATIONS[name])
        import sweep3
        with Pool(14) as pool:
            res = dict(pool.map(sweep3.work_kw, [(i, ABLATIONS[name]) for i in range(len(index()))], chunksize=8))
        json.dump(res, open(f'{SCR}/ablate_{name}.json', 'w'))
        A, B = corpus_decisions(res, Pa, use_partner=False)
        nprior = nany = 0
        pages_any, pages_prior = set(), set()
        for k, v in res.items():
            if 'err' in v or v['wi'] < 5:
                continue
            cs = F._uncrowded(v['c'], Pa) if Pa.crowd_in > 0 else v['c']
            for c in cs:
                if c['score'] >= 1.0:
                    nany += 1
                    pages_any.add(int(k))
                    if F._near_prior(c['y_frac'], Pa):
                        nprior += 1
                        pages_prior.add(int(k))
        # test document: relevant folds detected (single page, tier A rule) and false lines on its pages
        rel = {'p4': [0.502], 'p6': [0.5], 'p9': [0.337, 0.678], 'p11': [0.338, 0.68], 'p13': [0.339, 0.681], 'p14': [0.336]}
        det = tot = 0
        for pg, ys in rel.items():
            cs, (wi, hi) = F.line_candidates(td_page(int(pg[1:])), 300.0, Pa)
            fs = F.decide(cs, wi, hi, Pa, None)
            for y in ys:
                tot += 1
                det += any(abs(f.y_frac - y) <= 0.015 for f in fs)
        summary[name] = dict(corpus_lines_ge1_any_row=nany, corpus_pages_any_row=len(pages_any), corpus_lines_ge1_near_prior=nprior, corpus_pages_near_prior=len(pages_prior),
                             tierA_pages_decide=len(A), td_relevant_detected=f'{det}/{tot}')
        print(name, summary[name], flush=True)
    json.dump(summary, open('ablation_results.json', 'w'), indent=1)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'ablate':
        ablate(sys.argv[2:] or list(ABLATIONS))
    else:
        main()
