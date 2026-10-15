"""Try skew-voter variants on the r4 synthetic ground truth. .venv/bin/python bench/skew_variants.py"""
import os, sys, json, time
sys.dont_write_bytecode = True
os.environ["OMP_NUM_THREADS"] = "1"
import cv2, numpy as np
from multiprocessing import Pool
cv2.setNumThreads(1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R4 = os.path.join(ROOT, "research", "r4_geometry")
from scanpp import geometry as G
idx = json.load(open(f"{R4}/index.json")); R0 = json.load(open(f"{R4}/gt_results.json"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import skew_cand as C

def run(t):
    i, a = t
    im = cv2.imread(f"{R4}/{idx[i]['fn']}"); h, wd = im.shape[:2]
    r = cv2.warpAffine(im, cv2.getRotationMatrix2D((wd / 2, h / 2), a, 1.0), (wd, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    g = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    out = {}
    for name in C.VARIANTS:
        t0 = time.process_time()
        est = C.VARIANTS[name](g, 100.0)
        out[name] = (est, time.process_time() - t0)
    return out

if __name__ == "__main__":
    trials = [(r["i"], r["a"]) for r in R0]
    with Pool(int(os.environ.get("PROCS", 8))) as p:
        R = p.map(run, trials, chunksize=2)
    a = np.array([t[1] for t in trials]); inr = np.abs(a) <= 5
    base = None
    for name in C.VARIANTS:
        ang = np.array([r[name][0]["ens"][0] for r in R]); conf = np.array([r[name][0]["ens"][1] for r in R])
        e = np.abs(ang - a)
        k = inr
        line = f"{name:14s} ens |a|<=5: mean {e[k].mean():.4f} p95 {np.percentile(e[k],95):.4f} fail>.3 {(e[k]>.3).mean()*100:.2f}% acc {(conf[k]>=.6).mean():.3f}"
        if base is not None:
            line += f" | vs base: mean|d| {np.abs(ang-base)[k].mean():.4f} max {np.abs(ang-base)[k].max():.3f}"
        else: base = ang
        for v in ("proj", "lsd", "cc", "grad"):
            ev = np.array([r[name][0]["e"][v] for r in R])
            line += f" {v}:{np.abs(ev - a)[k].mean():.3f}"
        line += f" | cpu ms {np.mean([r[name][1] for r in R])*1000:.1f}"
        print(line)
