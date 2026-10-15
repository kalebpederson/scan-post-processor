"""cProfile of the whole per-page pipeline (process_array + encode): .venv/bin/python bench/prof_all.py [dpi]"""
import sys, cv2, cProfile, pstats
cv2.setNumThreads(1)
from pathlib import Path
from scanpp.options import Options
from scanpp.pipeline import process_array
dpi = int(sys.argv[1]) if len(sys.argv) > 1 else 300
imgs = []
for n in ("pair02_front", "pair08_front", "single02"):
    im = cv2.cvtColor(cv2.imread(f"research/corpus/{n}.jpg"), cv2.COLOR_BGR2RGB)
    if dpi != 300: im = cv2.resize(im, None, fx=dpi / 300, fy=dpi / 300, interpolation=cv2.INTER_AREA)
    imgs.append(im)
o = Options()
process_array(imgs[0].copy(), dpi, o, False)
pr = cProfile.Profile(); pr.enable()
for im in imgs: process_array(im.copy(), dpi, o, False)
pr.disable()
pstats.Stats(pr).sort_stats("tottime").print_stats(18)
