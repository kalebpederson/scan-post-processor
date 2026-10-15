"""Does faint gray haze (d=255-min in 1..25) left by the cleaner cost bytes? Compare encodings of the cleaned page vs the same page with haze set to white."""
from config import *
from encoders import *
from multiprocessing import Pool
CF = dict(E1_j150_q40_420=lambda r: enc_jpeg(r, r=150, q=40, sub=2), E1_j200_q55_420=lambda r: enc_jpeg(r, r=200, q=55, sub=2),
          E3_c75_q40=lambda r: enc_mrc(r, col_ppi=75, q=40), E4_s7=lambda r: enc_stencils(r, K=7), E2_p150_k8_km=lambda r: enc_palette(r, r=150, k=8))
def dehaze(rgb, T=25):
    d = 255 - rgb.min(-1).astype(np.int16)
    # remove only faint pixels that are not next to real ink (keeps antialiased edges)
    solid = (d > T).astype(np.uint8); near = cv2.dilate(solid, np.ones((5, 5), np.uint8)) > 0
    out = rgb.copy(); out[(d <= T) & (d > 0) & ~near] = 255; return out
def job(n):
    rgb = load_clean(n); d = 255 - rgb.min(-1).astype(np.int16); de = dehaze(rgb)
    r = dict(page=n, haze_frac=float(((d > 0) & (d <= 25)).mean()), removed_frac=float((rgb != de).any(-1).mean()), nonwhite=float((d > 0).mean()))
    for k, f in CF.items():
        a = f(rgb).payload(); b = f(de).payload(); r[k] = (a, b)
    return r
if __name__ == "__main__":
    info = json.load(open(WORK / "res" / "pages.json"))
    names = [i["name"] for i in info]
    with Pool(8) as p: res = p.map(job, names, chunksize=1)
    json.dump(res, open(WORK / "res" / "haze.json", "w"), indent=1)
    res.sort(key=lambda r: -r["removed_frac"])
    for r in res[:8] + res[-2:]:
        print(r["page"], "haze %.4f removed %.4f" % (r["haze_frac"], r["removed_frac"]), {k: "%d->%d (%+.1f%%)" % (v[0], v[1], 100 * (v[1] / v[0] - 1)) for k, v in r.items() if isinstance(v, tuple)})
    import numpy as np
    for k in CF: print(k, "mean saving over all pages: %.2f%%" % (100 * np.mean([1 - r[k][1] / r[k][0] for r in res])))
