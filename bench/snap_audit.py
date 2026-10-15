"""Audit of the neutral-ink snap: which reference strokes (research/r3_encoding/metrics2.py components of the BASE cleaner output) change
colour identity under the new cleaner, shown as ORIGINAL | before | after with the numbers that decide neutral-vs-coloured.
Usage: snap_audit.py --base old_clean.py --bucket 25,40 --n 48 --seed 5 --out sheet.png [--json meta.json]"""
import sys, os, json, argparse, importlib.util, random
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "research", "r3_encoding"))
ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--new", default=os.path.join(ROOT, "src/scanpp/clean.py"))
ap.add_argument("--bucket", default="25,40"); ap.add_argument("--n", type=int, default=48); ap.add_argument("--seed", type=int, default=5)
ap.add_argument("--out", required=True); ap.add_argument("--json"); ap.add_argument("--procs", type=int, default=8); ap.add_argument("--cols", type=int, default=4)
A = ap.parse_args()
import numpy as np, cv2
cv2.setNumThreads(1)
from multiprocessing import Pool
from PIL import Image, ImageDraw
LO, HI = (float(v) for v in A.bucket.split(","))
HALF = 40


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def analyse(p, want=None):
    """Returns (changed strokes meta list, tiles for the wanted cids)."""
    import config, metrics2
    from scipy import ndimage
    base, new = load(A.base, "bc"), load(A.new, "nc")
    src = config.to300(config.load_src(p), p["native"])
    ref = base.clean_page(src, 300, paper="flatten"); ren = new.clean_page(src, 300, paper="flatten")
    info = metrics2.RefInfo(p["name"], ref)
    if not len(info.ids):
        return [], []
    rl = metrics2._lab(ren[info.ys, info.xs])
    ra = np.asarray(ndimage.median(rl[:, 1], info.lab_px, info.ids)); rb = np.asarray(ndimage.median(rl[:, 2], info.lab_px, info.ids))
    dh = metrics2._hdiff(metrics2._hue(ra, rb), info.ref_h); rC = np.hypot(ra, rb)
    chg = (dh > 20) | (rC < 0.6 * info.ref_C)
    sel = [k for k in range(len(info.ids)) if chg[k] and LO <= info.ref_C[k] < HI]
    metas = [dict(page=p["name"], cid=int(info.ids[k]), refC=float(info.ref_C[k]), newC=float(rC[k]), dh=float(dh[k])) for k in sel]
    tiles = []
    if want:
        objs = ndimage.find_objects(info.L)
        bgs = base.estimate_background(src, 300, 0.35, 1.5, 0.8, 50.0, 0.06)
        for cid in want:
            k = int(np.nonzero(info.ids == cid)[0][0]); sl = objs[cid - 1]
            cy, cx = (sl[0].start + sl[0].stop) // 2, (sl[1].start + sl[1].stop) // 2
            y0 = int(np.clip(cy - HALF, 0, src.shape[0] - 2 * HALF)); x0 = int(np.clip(cx - HALF, 0, src.shape[1] - 2 * HALF))
            m = info.lab_px == cid
            raw = np.median(src[info.ys[m], info.xs[m]], 0)
            by, bx = min(bgs.shape[0] - 1, cy // 6), min(bgs.shape[1] - 1, cx // 6)
            paper = bgs[by, bx]
            q = np.clip(src[info.ys[m], info.xs[m]].astype(float) / np.maximum(paper, 1), 0, 1.5); q = np.minimum(q, 1)
            r = 1 - np.median(q, 0); dn = r.max(); chrom = r.max() - r.min()
            crop = lambda im: im[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF]
            tiles.append(dict(cid=cid, tiles=[crop(src), crop(ref), crop(ren)], paper=paper.astype(int).tolist(), raw=raw.astype(int).tolist(),
                              dens=float(dn), chrom=float(chrom), ratio=float(chrom / max(dn, 1e-3)),
                              bbox=(sl[1].start - x0, sl[0].start - y0, sl[1].stop - x0, sl[0].stop - y0)))
    return metas, tiles


def job1(p): return analyse(p)[0]
def job2(a): p, want = a; return p["name"], analyse(p, want)[1]


if __name__ == "__main__":
    import config
    pages = {p["name"]: p for p in config.pages()}
    with Pool(A.procs) as pool:
        allm = [m for ms in pool.map(job1, list(pages.values()), chunksize=1) for m in ms]
        print("changed strokes in bucket:", len(allm))
        rnd = random.Random(A.seed); pick = rnd.sample(allm, min(A.n, len(allm)))
        byp = {}
        for m in pick: byp.setdefault(m["page"], []).append(m["cid"])
        res = dict(pool.map(job2, [(pages[n], c) for n, c in byp.items()], chunksize=1))
    tiles = {}
    for n, tl in res.items():
        for t, cid in zip(tl, byp[n]): tiles[(n, cid)] = t
    W, Hh = 6 * 2 * HALF + 4, 2 * HALF * 2 + 44
    cols = A.cols; rows = (len(pick) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (3 * 2 * HALF + 12), rows * Hh), (235, 235, 235)); d = ImageDraw.Draw(sheet)
    out = []
    for i, m in enumerate(pick):
        t = tiles[(m["page"], m["cid"])]; x = (i % cols) * (3 * 2 * HALF + 12); y = (i // cols) * Hh
        trip = np.hstack([cv2.resize(a, None, fx=1, fy=1) for a in t["tiles"]])
        trip = np.repeat(np.repeat(trip, 1, 0), 1, 1)
        im = Image.fromarray(trip).resize((trip.shape[1] * 1, trip.shape[0] * 1), Image.NEAREST)
        im = im.resize((im.width, im.height * 1))
        sheet.paste(im.resize((im.width, im.height)), (x, y + 30))
        bx0, by0, bx1, by1 = t["bbox"]; d.rectangle((x + bx0, y + 30 + by0, x + bx1, y + 30 + by1), outline=(255, 0, 0))
        d.text((x + 2, y + 1), f"#{i} {m['page']} C {m['refC']:.0f}->{m['newC']:.0f} dh {m['dh']:.0f}", fill=(0, 0, 0))
        d.text((x + 2, y + 12), f"paper {t['paper']} raw {t['raw']}", fill=(0, 0, 0))
        d.text((x + 2, y + 21), f"rel.dens {t['dens']:.2f} chroma {t['chrom']:.2f} ratio {t['ratio']:.2f}", fill=(160, 0, 0))
        out.append(dict(i=i, **m, paper=t["paper"], raw=t["raw"], dens=t["dens"], chrom=t["chrom"], ratio=t["ratio"]))
    sheet.save(A.out)
    if A.json: json.dump(out, open(A.json, "w"), indent=1)
    print("saved", A.out, sheet.size)
