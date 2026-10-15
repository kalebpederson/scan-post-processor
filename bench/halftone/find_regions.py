"""Candidate finder for light halftoned print: on each page compute native ink density d (flat-field), dm = d blurred at the
halftone pitch (sigma 0.006 in).  A 0.5 in tile is a candidate if it has no strong ink (max dm < 0.22) but a lot of fine-scale
structure: fraction of pixels with d - dm > 0.10 (isolated dark dots / thin strokes) >= 1.5%.  Writes candidates.json (tiles) for
visual review (see make_regions.py)."""
import sys, os, json, glob
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
import cv2, numpy as np
from scanpp import clean as C
cv2.setNumThreads(2)

def density(rgb, dpi):
    P = C.DEFAULTS
    bgs = np.maximum(C.estimate_background(rgb, dpi, P["bg_win_in"], P["guard_win_in"], P["blob_guard"], P["bg_work_dpi"], P["bg_open_in"]), 16)
    h, w = rgb.shape[:2]
    q = [rgb[..., c] / cv2.resize(np.ascontiguousarray(bgs[..., c]), (w, h)) for c in range(3)]
    return np.clip(1 - np.min(q, 0), 0, 1).astype(np.float32)

def cands(rgb, dpi):
    d = density(rgb, dpi); dm = cv2.GaussianBlur(d, (0, 0), max(0.6, 0.006 * dpi))
    t = int(0.5 * dpi); out = []
    fine = ((d - dm) > 0.10).astype(np.float32)
    H, W = d.shape
    for y in range(int(.03 * H), H - t, t // 2):
        for x in range(int(.05 * W), W - t, t // 2):
            mx = dm[y:y + t, x:x + t].max()
            if mx >= 0.22 or mx < 0.06: continue
            fr = float(fine[y:y + t, x:x + t].mean())
            if fr >= 0.015: out.append((x, y, t, round(fr, 4), round(float(mx), 3)))
    return out

if __name__ == "__main__":
    res = {}
    man = json.load(open(os.path.join(ROOT, "research/corpus/manifest.json")))
    for m in man:
        f = m["file"]
        if f.startswith("single26"): continue
        rgb = cv2.cvtColor(cv2.imread(os.path.join(ROOT, "research/corpus", f)), cv2.COLOR_BGR2RGB)
        dpi = float(m["dpi"][0]); c = cands(rgb, dpi)
        res[f] = dict(dpi=dpi, tiles=c); print(f, dpi, len(c), flush=True)
    json.dump(res, open(sys.argv[1], "w"))
