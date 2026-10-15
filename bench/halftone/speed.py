"""Per-page clean_page CPU time at 300 dpi (single thread, best of N process-CPU) with the halftone handling on vs off
(desc_sigma_in=0 == previous cleaner).  Usage: speed.py [reps]"""
import sys, os, time
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.join(ROOT, "src"))
import cv2; cv2.setNumThreads(1)
from scanpp.clean import clean_page
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
pages = ["pair02_front", "pair08_front", "single02", "single03", "image02"]
ims = [cv2.cvtColor(cv2.imread(f"{ROOT}/research/corpus/{n}.jpg"), cv2.COLOR_BGR2RGB) for n in pages]
res = {}
for tag, P in (("old (off)", {"desc_sigma_in": 0}), ("new", {}), ("old (off)", {"desc_sigma_in": 0}), ("new", {})):
    tot = 0.0
    for im in ims:
        clean_page(im, 300, **P); best = 1e9
        for _ in range(reps):
            t = time.process_time(); clean_page(im, 300, **P); best = min(best, time.process_time() - t)
        tot += best
    res.setdefault(tag, []).append(tot / len(ims))
for k, v in res.items(): print(f"{k}: " + ", ".join(f"{x*1000:.0f} ms/page" for x in v))
