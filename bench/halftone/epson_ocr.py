"""Metric (a): OCR word recall on the Epson keymap, restricted to R1's lightest darkness class.
Uses R1's reference words (research/r1_resolution/work/consensus.json, 600-dpi grid, page rotated 90 CW), R1's tesseract
settings (psm 11, oem 1, --dpi 300), the cleaned page at 300 dpi gray (no encoding), and R1's positional scorer.
Usage: epson_ocr.py [--clean NAME] -> prints recall for all/light/lightest classes, raw vs cleaned."""
import os, sys, json
sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R1 = os.path.join(ROOT, "research", "r1_resolution")
sys.path.insert(0, R1); sys.path.insert(0, os.path.join(ROOT, "src"))
import cv2, numpy as np
cv2.setNumThreads(4)
os.environ.setdefault("R1_TMP", os.environ.get("TMPDIR", "/tmp"))
import epson_eval as E      # R1 scorer (reads consensus.json at import)

def epson300():
    im = cv2.imread(os.path.join(ROOT, "research/scans/epson_300dpi.png"), cv2.IMREAD_COLOR)
    return cv2.cvtColor(cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE), cv2.COLOR_BGR2RGB)

def score_rgb(rgb, dpi=300):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hyp = E.ocr(g, dpi, 11)
    s = E.score(hyp, 600.0 / dpi)
    hits = np.array(s["hits"], bool)
    dark = np.array([w["dark"] for w in E.REF])
    out = dict(n_hyp=s["n_hyp"], pos_all=s["pos_recall"], ms_all=s["ms_recall"], pos_light=s["pos_light"])
    for name, m in (("lightest25", dark < 25), ("lightest30", dark < 30), ("light_ge30", (dark >= 30) & (dark < 35))):
        out["pos_" + name] = float(hits[m].mean()); out["n_" + name] = int(m.sum())
    out["hits"] = hits.tolist()
    return out

if __name__ == "__main__":
    from scanpp.clean import clean_page
    rgb = epson300()
    for name, im in (("raw", rgb), ("clean", clean_page(rgb, 300))):
        r = score_rgb(im); r.pop("hits")
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
