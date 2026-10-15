"""Shared config. To re-run the whole study with a different cleaner:
   CLEANER_DIR=/path/to/dir_containing_clean.py ./run_all.sh       (dir must expose clean.clean_page(rgb, dpi, paper="flatten"))
Work products are cached under work/<cleaner-tag>/ so different cleaners do not collide."""
import os, sys, json, importlib
from pathlib import Path
import numpy as np, cv2
cv2.setNumThreads(1)
HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parent
CORPUS = RESEARCH / "corpus"
SCANS = RESEARCH / "scans"
R1 = RESEARCH / "r1_resolution"
CLEANER_DIR = Path(os.environ.get("CLEANER_DIR", HERE / "clean_snapshot")).resolve()
TAG = os.environ.get("CLEANER_TAG", CLEANER_DIR.name)
WORK = HERE / "work" / TAG
for d in ("clean", "pdf", "render", "res", "tmp"): (WORK / d).mkdir(parents=True, exist_ok=True)
TMP = str(WORK / "tmp")

def load_cleaner():
    sys.path.insert(0, str(CLEANER_DIR))
    import clean  # noqa
    return clean.clean_page
def clean_rgb(rgb, dpi): return load_cleaner()(rgb, dpi, paper="flatten")

def pages():
    """list of dict(name, native_dpi, kind). dims are all ~2544x3295 for 300 dpi-equivalent."""
    out = []
    for m in json.load(open(CORPUS / "manifest.json")):
        d = int(round(m["dpi"][0]))
        if m["file"] == "single26.png": d = 300          # manifest dpi 833 is bogus metadata; 2544x3295 px page
        out.append(dict(name=Path(m["file"]).stem, file=str(CORPUS / m["file"]), native=d, kind="corpus"))
    out.append(dict(name="epson300", file=str(SCANS / "epson_300dpi.png"), native=300, kind="epson"))
    return out

def load_src(p):
    im = cv2.imread(p["file"], cv2.IMREAD_COLOR)
    if p["kind"] == "epson": im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    return im[:, :, ::-1].copy()   # RGB

def to300(rgb, native):
    if native == 300: return rgb
    h, w = rgb.shape[:2]
    return cv2.resize(rgb, (round(w * 300 / native), round(h * 300 / native)), interpolation=cv2.INTER_CUBIC)

def clean_path(name): return WORK / "clean" / f"{name}.png"
def load_clean(name): return cv2.imread(str(clean_path(name)), cv2.IMREAD_COLOR)[:, :, ::-1].copy()
