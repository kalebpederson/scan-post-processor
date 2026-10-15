"""Duplex-pair discovery + verification.
For every source PDF of the corpus (extracted read-only with pdfimages), test every unordered page pair (i,j) with equal size:
mirror one horizontally, search translation (+-1.3in) on 'faint layer' (A, real ink masked) vs 'ink' (mirrored B) with normalised cross-corr,
in both directions (A bleeds from B, B bleeds from A). Verified if peak z-score >= ZMIN and peak not at search border; refine with ECC.
Writes work/pages/<slug>/pg-NNN.jpg and pairs2.json."""
import sys, json, subprocess, hashlib; sys.path.insert(0, '.')
from common import *
from multiprocessing import Pool
WS = 100.0; ZMIN = 7.0; PAGES = WORK / "pages"

def slug(src): return hashlib.md5(src.encode()).hexdigest()[:8]

def extract(src):
    d = PAGES / slug(src); d.mkdir(parents=True, exist_ok=True)
    if not list(d.glob("pg-*.jpg")):
        subprocess.run(["pdfimages", "-j", str(ROOT / src), str(d / "pg")], check=True)
    return d

def dm(path, dpi):
    r = load_rgb(path); d, _ = darkness(r, dpi); s = dpi / WS
    return cv2.resize(d, (int(r.shape[1] / s), int(r.shape[0] / s)), interpolation=cv2.INTER_AREA), r.shape[:2]

def test_dir(A, B):
    """A receives bleed from mirrored B. returns (z, peak, dx, dy, ncc) shift s.t. A(x,y) ~ Bm(x+dx, y+dy)"""
    h, w = A.shape
    real = cv2.dilate((A > 0.25).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    Fm = np.where(real, 0, np.clip(A, 0, .25)).astype(np.float32)
    Fm[:, :int(.1 * w)] = 0; Fm[:, int(.9 * w):] = 0
    Bm = np.ascontiguousarray(np.clip(B[:, ::-1], 0, .6)).astype(np.float32)
    R = int(1.3 * WS)
    Bp = cv2.copyMakeBorder(Bm, R, R, R, R, cv2.BORDER_CONSTANT, value=0)
    T = np.ascontiguousarray(Fm[R:h - R, R:w - R])
    res = cv2.matchTemplate(Bp, T, cv2.TM_CCOEFF_NORMED)
    i = np.unravel_index(res.argmax(), res.shape)
    z = (res.max() - res.mean()) / res.std()
    border = i[0] in (0, res.shape[0] - 1) or i[1] in (0, res.shape[1] - 1)
    # res index (iy,ix) = offset of template in Bp -> template origin at (R,R) in A coords ... shift = i - 2R? Bp padded by R, template origin R => shift = i - R - R + R
    dy = i[0] - R; dx = i[1] - R   # A(x,y) <-> Bm(x+dx - ... ) (template cropped at R)  => A(x,y)~Bm(x+dx, y+dy)
    return float(z), float(res.max()), int(dx), int(dy), bool(border), Bm, Fm

def do_pdf(src):
    d = extract(src)
    pgs = sorted(d.glob("pg-*.jpg"))
    if len(pgs) > 6: pgs = pgs[:6]
    dpi = None
    for e in manifest():
        if e["source"] == src: dpi = dpi_of(e); break
    maps_ = []
    for p in pgs:
        m, shp = dm(p, dpi); maps_.append((m, shp))
    out = []
    for i in range(len(pgs)):
        for j in range(i + 1, len(pgs)):
            if maps_[i][1] != maps_[j][1]: continue
            for a, b in ((i, j), (j, i)):
                z, pk, dx, dy, border, Bm, Fm = test_dir(maps_[a][0], maps_[b][0])
                out.append(dict(src=src, dpi=dpi, front=a + 1, back=b + 1, front_file=str(pgs[a]), back_file=str(pgs[b]),
                                z=round(z, 1), peak=round(pk, 3), dx100=dx, dy100=dy, border=border))
    return out

if __name__ == "__main__":
    srcs = sorted({e["source"] for e in manifest() if e["source"].endswith(".pdf")})
    with Pool(16) as p: res = p.map(do_pdf, srcs)
    flat = [r for x in res for r in x]
    json.dump(flat, open("pairs2_raw.json", "w"), indent=1)
    good = [r for r in flat if r["z"] >= ZMIN and not r["border"]]
    print(len(flat), "directed candidates;", len(good), "pass z>=%g" % ZMIN)
    for r in sorted(flat, key=lambda r: -r["z"])[:45]:
        print(r["z"], r["peak"], r["front"], "<-", r["back"], (r["dx100"], r["dy100"]), r["border"], r["src"][-40:])
