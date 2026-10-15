"""Colour-identity regression of a cleaner change: research/r3_encoding/metrics2.py (stroke colour identity per component,
highlighter fringe, pencil weight) with the reference = the page cleaned by the BASE cleaner and the rendering under test = the page
cleaned by the NEW cleaner (both from the r2/r3 corpus pages at 300 ppi, no encoding).  A base-vs-base run is trivially 0.
Usage: colorid.py --base path/to/clean.py [--new path/to/clean.py] [--out FILE.json] [--procs 8]"""
import sys, os, json, argparse, importlib.util
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R3 = os.path.join(ROOT, "research", "r3_encoding")
sys.path.insert(0, R3)
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--new", default=os.path.join(ROOT, "src/scanpp/clean.py"))
ap.add_argument("--params", default="{}", help="JSON kwargs for the new clean_page"); ap.add_argument("--out"); ap.add_argument("--procs", type=int, default=8); A = ap.parse_args()
import numpy as np, cv2
cv2.setNumThreads(1)
from multiprocessing import Pool


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def job(p):
    import config, metrics2
    base, new = load(A.base, "base_clean"), load(A.new, "new_clean")
    src = config.to300(config.load_src(p), p["native"])
    ref = base.clean_page(src, 300, paper="flatten"); ren = new.clean_page(src, 300, paper="flatten", **json.loads(A.params))
    info = metrics2.RefInfo(p["name"], ref)
    r = metrics2.evaluate(info, ren)
    if len(info.ids):       # per-stroke breakdown by reference chroma (metrics2 logic, kept per component)
        from scipy import ndimage
        rl = metrics2._lab(ren[info.ys, info.xs])
        ra = np.asarray(ndimage.median(rl[:, 1], info.lab_px, info.ids)); rb = np.asarray(ndimage.median(rl[:, 2], info.lab_px, info.ids))
        dh = metrics2._hdiff(metrics2._hue(ra, rb), info.ref_h); rC = np.hypot(ra, rb)
        r["strokes"] = [[float(c), bool((h > 20) or (n < 0.6 * c)), float(a)] for c, h, n, a in zip(info.ref_C, dh, rC, info.area[info.ids])]
    r["name"] = p["name"]; r["diff_mean"] = float(np.abs(ref.astype(int) - ren).mean())
    return r


if __name__ == "__main__":
    import config
    with Pool(A.procs) as pool:
        res = pool.map(job, config.pages(), chunksize=1)
    if A.out: json.dump(res, open(A.out, "w"), indent=1)
    have = [r for r in res if r.get("cid_n")]
    n = sum(r["cid_n"] for r in have)
    w = lambda k: sum(r[k] * r["cid_n"] for r in have if r.get(k) is not None) / n
    print(f"pages {len(res)}  strokes {n}  changed {w('cid_changed'):.4f}  hue>20deg {w('cid_hue'):.4f}  chroma<0.6 {w('cid_chroma'):.4f}  area-weighted {w('cid_area_changed'):.4f}  swap {w('cid_swap'):.4f}")
    for k in ("hl_ink", "hl_near", "hl_hl", "pencil_ratio"):
        v = [r[k] for r in res if r.get(k) is not None]
        if v: print(f"  {k}: mean {np.mean(v):.3f} over {len(v)} pages")
    st = [x for r in res for x in r.get("strokes", [])]
    for lo, hi in ((25, 40), (40, 60), (60, 999)):
        sel = [x for x in st if lo <= x[0] < hi]
        print(f"  ref chroma [{lo},{hi}): {len(sel)} strokes, failure rate {np.mean([x[1] for x in sel]):.4f}")
    print("  mean abs pixel diff base vs new:", round(float(np.mean([r['diff_mean'] for r in res])), 4))
