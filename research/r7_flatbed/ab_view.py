"""usage: ab_view.py NAME OUT_DIR [--sample K] [--seed S]
For the (input, page) pairs whose pipeline output really differs between the previous code and the working tree (see
ab_pdfcmp.py; images count as page 0), write before|after|diff crops to OUT_DIR (local only).  PDF pages are rasterised
at 150 dpi.  With --sample K only K random pairs plus the 10 largest differences are rendered.  Also writes
OUT_DIR/stats.json: per rendered pair the changed-pixel fraction of the page (local use)."""
import json
import random
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

H = Path(__file__).resolve().parent
name, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
summ = json.load(open(H / f"ab_{name}.json"))
base = H / "work" / f"ab_{name}"


def raster(path, k, tmp):
    if path.suffix == ".pdf":
        stem = Path(tmp) / f"{path.parent.name}_{path.stem}_{k}"
        subprocess.run(["pdftoppm", "-r", "150", "-f", str(k + 1), "-l", str(k + 1), "-png", "-singlefile", str(path), str(stem)], check=True)
        return cv2.imread(str(stem) + ".png")
    return cv2.imread(str(path))


def one(pair):
    i, k = pair
    o = next(iter((base / "old").glob(f"{i:04d}.*")), None)
    if o is None:
        return pair, None
    n = base / "new" / o.name
    with tempfile.TemporaryDirectory(dir=H / "work") as tmp:
        a, b = raster(o, k, tmp), raster(n, k, tmp)
    if a is None or b is None or a.shape != b.shape:
        return pair, None
    d = (a != b).any(axis=2)
    if not d.any():
        return pair, 0.0
    ys, xs = np.nonzero(d)
    pad = 40
    y0, y1 = max(0, ys.min() - pad), min(a.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(a.shape[1], xs.max() + 1 + pad)
    y1, x1 = min(y1, y0 + 700), min(x1, x0 + 700)
    hl = a[y0:y1, x0:x1].copy()
    hl[d[y0:y1, x0:x1]] = (0, 0, 255)
    sep = np.full((y1 - y0, 4, 3), (255, 0, 0), np.uint8)
    cv2.imwrite(str(out / f"f{i:04d}_p{k}.png"), np.hstack([a[y0:y1, x0:x1], sep, b[y0:y1, x0:x1], sep, hl]))
    return pair, float(d.mean())


def main():
    pairs = [tuple(p) for p in summ["real_pairs"]] + [(i, 0) for i in summ["inputs_with_real_page_differences"]
                                                          if not any(p[0] == i for p in summ["real_pairs"])]
    if "--sample" in sys.argv:
        kk = int(sys.argv[sys.argv.index("--sample") + 1])
        seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
        pairs = random.Random(seed).sample(pairs, min(kk, len(pairs)))
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(one, pairs))
    stats = {f"{p[0]}_{p[1]}": v for p, v in res}
    (out / "stats.json").write_text(json.dumps(stats))
    vals = sorted(v for v in stats.values() if v)
    print(len(pairs), "pairs rendered;", "changed-pixel fraction median %.5f max %.5f" % (vals[len(vals) // 2], vals[-1]) if vals else "")


if __name__ == "__main__":
    main()
