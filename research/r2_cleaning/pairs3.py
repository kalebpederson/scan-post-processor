"""Refine registration of verified sheets (from pairs2_raw.json) with ECC affine at 100 dpi, write pairs_verified.json.
For each *directed* instance: front A receives bleed from mirrored back B.  warp (full-res 2x3) is for cv2.warpAffine(Bm_ink, warp,
(w,h), flags=INTER_LINEAR|WARP_INVERSE_MAP)  [maps A coords -> Bm coords]."""
import sys, json; sys.path.insert(0, '.')
from common import *
from pairs2 import WS, ZMIN, dm
from multiprocessing import Pool
R = int(1.3 * WS)

def refine(r):
    dpi = r["dpi"]
    A, shp = dm(r["front_file"], dpi); B, _ = dm(r["back_file"], dpi)
    if B.shape != A.shape: B = cv2.resize(B, (A.shape[1], A.shape[0]))
    h, w = A.shape
    real = cv2.dilate((A > 0.25).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    F = np.where(real, 0, np.clip(A, 0, .25)).astype(np.float32)
    Bm = np.ascontiguousarray(np.clip(B[:, ::-1], 0, .6)).astype(np.float32)
    valid = (~real).astype(np.uint8); valid[:, :int(.1 * w)] = 0; valid[:, int(.9 * w):] = 0
    Fb = cv2.GaussianBlur(F, (0, 0), 1.2); Bb = cv2.GaussianBlur(Bm, (0, 0), 1.2)
    warp = np.array([[1, 0, r["sx100"]], [0, 1, r["sy100"]]], np.float32)
    # sign check: template A(x,y) ~ Bm(x+sx, y+sy)  => input coords = warp @ [x,y,1]
    def ncc(wp):
        al = cv2.warpAffine(Bb, wp, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        m = valid > 0; a = Fb[m] - Fb[m].mean(); b = al[m] - al[m].mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-12))
    # the matchTemplate offset convention was validated empirically: choose sign by ncc
    cands = [np.array([[1, 0, sx], [0, 1, sy]], np.float32) for sx, sy in ((r["sx100"], r["sy100"]), (-r["sx100"], -r["sy100"]))]
    scores = [ncc(c) for c in cands]; warp = cands[int(np.argmax(scores))]; n0 = max(scores)
    rho = None
    try:
        rho, warp2 = cv2.findTransformECC(Fb, Bb, warp.copy(), cv2.MOTION_AFFINE,
                                          (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), valid, 3)
        n1 = ncc(warp2)
        if n1 >= n0: warp = warp2; n0 = n1
    except cv2.error:
        pass
    s = dpi / WS; Wf = warp.copy(); Wf[:, 2] *= s
    r = dict(r); r.update(ncc_after=round(n0, 3), ecc_rho=None if rho is None else round(float(rho), 3), warp_full=Wf.tolist(),
                         shift_in=[round(float(Wf[0, 2]) / dpi, 3), round(float(Wf[1, 2]) / dpi, 3)],
                         lin=[round(float(Wf[0, 0]), 4), round(float(Wf[0, 1]), 4), round(float(Wf[1, 0]), 4), round(float(Wf[1, 1]), 4)])
    return r

if __name__ == "__main__":
    raw = json.load(open("pairs2_raw.json"))
    sheets = {}
    for r in raw:
        r["sx100"] = r["dx100"] - R; r["sy100"] = r["dy100"] - R
        if r["z"] >= ZMIN and not r["border"] and abs(r["sx100"]) <= 60 and abs(r["sy100"]) <= 60:
            sheets.setdefault((r["src"], min(r["front"], r["back"]), max(r["front"], r["back"])), []).append(r)
    # for verified sheets include both directions (if the other direction was not listed as passing, take it from raw)
    todo = []
    for (src, a, b), v in sheets.items():
        for ff, bb in ((a, b), (b, a)):
            cand = [r for r in raw if r["src"] == src and r["front"] == ff and r["back"] == bb][0]
            todo.append(cand)
    with Pool(16) as p: res = p.map(refine, todo)
    json.dump(res, open("pairs_verified.json", "w"), indent=1)
    for r in sorted(res, key=lambda r: (r["src"], r["front"])):
        print(r["src"][-38:], r["front"], "<-", r["back"], "z", r["z"], "ncc", r["ncc_after"], "rho", r["ecc_rho"], "shift_in", r["shift_in"], r["lin"])
