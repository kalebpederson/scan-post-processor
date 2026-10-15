"""Per-stage CPU benchmark.  Usage: .venv/bin/python bench/bench.py [--reps N]
Runs process_array (+ the JPEG encoder) on native-300 corpus pages (and the same pages area-resized to 200 dpi),
single-threaded, and prints process-CPU seconds per stage per page.  To compare with an older checkout run it with
PYTHONPATH=<old>/src.  CPU times inflate when the machine is loaded (SMT / cache sharing): check `uptime`."""
import argparse, sys, time
from pathlib import Path
import cv2
import numpy as np
cv2.setNumThreads(1)
from scanpp import geometry, clean
from scanpp.options import Options
from scanpp.pipeline import process_array
from scanpp.encode import encode_page

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["pair02_front", "pair08_front", "single02", "single03"]


def load(name, dpi):
    im = cv2.cvtColor(cv2.imread(str(ROOT / "research/corpus" / f"{name}.jpg")), cv2.COLOR_BGR2RGB)
    if dpi != 300:
        s = dpi / 300
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return im


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--reps", type=int, default=5); a = ap.parse_args()
    opts = Options()
    for dpi in (200, 300):
        imgs = [load(n, dpi) for n in PAGES]
        tot: dict = {}      # per stage: sum over pages of the best (min) CPU time over the reps (robust to machine load)
        mean: dict = {}
        n = 0
        for im in imgs:
            process_array(im.copy(), dpi, opts, allow_drop=False)       # warm up
            best: dict = {}
            for _ in range(a.reps):
                out, odpi, st = process_array(im.copy(), dpi, opts, allow_drop=False)   # (edge cleaning paints in place)
                t0 = time.process_time()
                encode_page(out, int(round(odpi)), (612, 792), opts.encoder)
                st["t"]["encode"] = time.process_time() - t0
                for k, v in st["t"].items():
                    best[k] = min(best.get(k, 1e9), v)
                    mean[k] = mean.get(k, 0) + v
                n += 1
            for k, v in best.items():
                tot[k] = tot.get(k, 0) + v
        print(f"--- {dpi} dpi, CPU s/page (best of {a.reps} per page, averaged over {len(imgs)} pages; [mean of all runs])")
        for k, v in tot.items():
            print(f"  {k:8s} {v / len(imgs):.3f}   [{mean[k] / n:.3f}]")
        print(f"  {'TOTAL':8s} {sum(tot.values()) / len(imgs):.3f}   (excl. encode: {sum(v for k, v in tot.items() if k != 'encode') / len(imgs):.3f})")


if __name__ == "__main__":
    main()
