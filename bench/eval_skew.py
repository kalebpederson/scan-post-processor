"""Skew ground-truth + blank-score gates against research/r4_geometry data, using scanpp.geometry.
Usage: .venv/bin/python bench/eval_skew.py OUT.json [--blank]
Skew: same 300 synthetic rotations as r4 gt_final.py (cache page rotated by known angle, estimate at 100 ppi).
Blank: package blank_score on all 1969 cached pages (dpi from index.json) -> compare with research blank_scores.json."""
import os, sys, json
sys.dont_write_bytecode = True
os.environ["OMP_NUM_THREADS"] = "1"
import cv2, numpy as np
from multiprocessing import Pool
cv2.setNumThreads(1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R4 = os.path.join(ROOT, "research", "r4_geometry")
from scanpp import geometry as G
idx = json.load(open(f"{R4}/index.json")); R0 = json.load(open(f"{R4}/gt_results.json"))


def w(t):
    i, a = t
    im = cv2.imread(f"{R4}/{idx[i]['fn']}"); h, wd = im.shape[:2]
    r = cv2.warpAffine(im, cv2.getRotationMatrix2D((wd / 2, h / 2), a, 1.0), (wd, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    ang, conf = G.estimate_skew(g, 100)
    return dict(i=i, a=a, est=ang, conf=conf)


def wbl(i):
    g, d = G.prep_gray(cv2.imread(f"{R4}/{idx[i]['fn']}", 0), 100)
    return G.blank_score(g, d)


if __name__ == "__main__":
    out = sys.argv[1]
    trials = [(r["i"], r["a"]) for r in R0]
    with Pool(int(os.environ.get("PROCS", 8))) as p:
        R = p.map(w, trials, chunksize=2)
        res = {}
        a = np.array([r["a"] for r in R]); est = np.array([r["est"] for r in R]); conf = np.array([r["conf"] for r in R])
        for name, m in (("|a|<=5", np.abs(a) <= 5), ("|a|<=2.5", np.abs(a) <= 2.5), ("5<|a|", np.abs(a) > 5)):
            for nm2, acc in (("all", np.ones(len(a), bool)), ("conf>=0.6", conf >= 0.6)):
                k = m & acc
                if k.sum() == 0: continue
                e = np.abs(est - a)[k]
                res[f"{name} {nm2}"] = dict(n=int(k.sum()), mean=float(e.mean()), p95=float(np.percentile(e, 95)), max=float(e.max()), fail03=float((e > 0.3).mean()))
        k = np.abs(a) <= 5
        res["accepted_frac"] = float((conf[k] >= 0.6).mean())
        res["raw"] = [[float(x), float(y), float(z)] for x, y, z in zip(a, est, conf)]
        if "--blank" in sys.argv:
            bs = np.array(p.map(wbl, range(len(idx)), chunksize=8))
            lab = json.load(open(f"{R4}/labels_blank.json"))["labels"]
            blank = np.array([lab.get(str(i), {}).get("label") == "BLANK" for i in range(len(idx))])
            ref = np.array(json.load(open(f"{R4}/blank_scores.json")))
            res["blank"] = dict(thr=G.BLANK_THRESHOLD, false_blank=int(((bs < G.BLANK_THRESHOLD) & ~blank).sum()),
                                missed_blank=int(((bs >= G.BLANK_THRESHOLD) & blank).sum()),
                                nearest_nonblank=float(bs[~blank].min()), max_blank=float(bs[blank].max()),
                                max_rel_diff_vs_research=float(np.max(np.abs(bs - ref) / np.maximum(ref, 1))),
                                flips_vs_research=int(((bs < 40) != (ref < 40)).sum()))
            res["blank_scores"] = bs.tolist()
    json.dump(res, open(out, "w"))
    for k, v in res.items():
        if k not in ("raw", "blank_scores"): print(k, v)
