"""Step 1: upsample sub-300 pages to 300, clean with the snapshot cleaner, cache; classify pages (blank / no-colour)."""
import time, sys
from multiprocessing import Pool
from config import *
from skimage.color import rgb2lab

def job(p):
    t = time.perf_counter()
    src = to300(load_src(p), p["native"])
    c = clean_rgb(src, 300)
    tc = time.perf_counter() - t
    cv2.imwrite(str(clean_path(p["name"])), c[:, :, ::-1])
    return dict(name=p["name"], native=p["native"], kind=p["kind"], t_clean=tc, **classify(c))

def classify(c):
    mn = c.min(-1).astype(np.int16); mx = c.max(-1).astype(np.int16)
    ink = mn < 255 - 64            # clearly inked pixels
    nonwhite = (c != 255).any(-1)
    n = c.shape[0] * c.shape[1]
    sat = (mx - mn) > 40
    colored = ink & sat
    # connected colour area: ignore specks (< 40 px, ~ half a stroke width x length) via opening count
    cc = cv2.connectedComponentsWithStats(colored.astype(np.uint8), connectivity=8)[2][1:, cv2.CC_STAT_AREA]
    color_px = int(cc[cc >= 40].sum()) if len(cc) else 0
    return dict(ink_frac=float(ink.mean()), nonwhite_frac=float(nonwhite.mean()), color_px=color_px,
                color_frac_of_ink=float(color_px / max(1, ink.sum())), h=c.shape[0], w=c.shape[1])

if __name__ == "__main__":
    P = pages()
    with Pool(16) as pool: res = pool.map(job, P, chunksize=1)
    # classes. blank: almost no ink. nocolor: <0.5% of ink px (and <1500 px) are saturated.
    for r in res:
        r["blank"] = bool(r["ink_frac"] < 4e-4)
        r["nocolor"] = bool(r["color_px"] < 1500 or r["color_frac_of_ink"] < 0.005)
        r["cls"] = "blank" if r["blank"] else ("gray" if r["nocolor"] else "color")
    json.dump(res, open(WORK / "res" / "pages.json", "w"), indent=1)
    for r in res: print(r["name"], r["native"], r["cls"], "ink %.4f color_px %d (%.3f) nonwhite %.4f  clean %.1fs" % (r["ink_frac"], r["color_px"], r["color_frac_of_ink"], r["nonwhite_frac"], r["t_clean"]))
