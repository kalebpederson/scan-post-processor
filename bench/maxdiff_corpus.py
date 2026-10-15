"""Max abs diff of the package clean_page vs a reference implementation over the whole R2 corpus (both paper modes).
Usage: .venv/bin/python bench/maxdiff_corpus.py [reference.py]   (default: research/r2_cleaning/clean.py)"""
import sys, json, importlib.util, os
sys.dont_write_bytecode = True
import cv2, numpy as np
from multiprocessing import Pool
cv2.setNumThreads(1)
ref_path = sys.argv[1] if len(sys.argv) > 1 else "research/r2_cleaning/clean.py"
def load():
    spec = importlib.util.spec_from_file_location("ref_clean", ref_path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
from scanpp import clean as new
man = json.load(open("research/corpus/manifest.json"))
def work(x):
    ref = load()
    im = cv2.cvtColor(cv2.imread("research/corpus/" + x["file"]), cv2.COLOR_BGR2RGB)
    out = []
    for paper in ("flatten", "keep"):
        d = np.abs(ref.clean_page(im, x["dpi"][0], paper=paper).astype(int) - new.clean_page(im, x["dpi"][0], paper=paper))
        out.append((int(d.max()), int((d > 0).sum()), int((d > 1).sum())))
    return x["file"], out
if __name__ == "__main__":
    with Pool(int(os.environ.get("PROCS", 12))) as p:
        R = p.map(work, [x for x in man if x["file"].endswith(".jpg")], chunksize=1)
    mx = max(max(o[0] for o in r[1]) for r in R)
    print("pages", len(R), "max abs diff", mx, "pixels differing (any):", sum(o[1] for r in R for o in r[1]), "differing by >1:", sum(o[2] for r in R for o in r[1]))
    print("worst:", sorted(R, key=lambda r: -max(o[0] for o in r[1]))[:3])
