"""Shared helpers for the r2_cleaning experiments (I/O, baseline B0, ink/paper maps, labelled regions)."""
import json, os, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CORPUS = ROOT / "research" / "corpus"
SCANS = ROOT / "research" / "scans"
WORK = HERE / "work"
WORK.mkdir(exist_ok=True)
cv2.setNumThreads(1)

def manifest():
    return json.load(open(CORPUS / "manifest.json"))

def dpi_of(entry):
    return int(round(entry["dpi"][0]))

def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))

def save_rgb(path, rgb):
    Image.fromarray(rgb).save(path)

def corpus_items():
    """list of (name, path, dpi) for all 56 corpus images + epson 300"""
    out = []
    for e in manifest():
        out.append((Path(e["file"]).stem, str(CORPUS / e["file"]), dpi_of(e)))
    return out

# ---------------------------------------------------------------- baseline B0
def b0(rgb, black=0.05098, white=0.866667, gamma=0.8822):
    """ImageMagick  -level 5.098%,86.6667%,0.8822  (per channel, global).
    magick: out = ((x-b)/(w-b)) ** (1/gamma)  (gamma>1 brightens). Verified vs magick in verify_b0()."""
    x = rgb.astype(np.float32) / 255.0
    y = np.clip((x - black) / (white - black), 0, 1) ** (1.0 / gamma)
    return np.clip(y * 255 + 0.5, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------- measurement-side maps (independent of the cleaners)
def lum(rgb_f):
    return rgb_f[..., 0] * 0.2126 + rgb_f[..., 1] * 0.7152 + rgb_f[..., 2] * 0.0722

def ref_background(rgb, dpi):
    """Reference (measurement) background: per-channel large percentile-ish estimate. Deliberately a different
    implementation from clean.py (median of a 1/8 downsample -> 90th pct filter), used for labelling/metrics."""
    s = max(1, int(round(dpi / 50)))
    small = cv2.resize(rgb, (rgb.shape[1] // s, rgb.shape[0] // s), interpolation=cv2.INTER_AREA)
    k = max(3, int(round(0.35 * 50)) | 1)
    bg = np.empty(small.shape, np.float32)
    for c in range(3):
        ch = small[..., c]
        mx = cv2.dilate(ch, np.ones((k, k), np.uint8))          # local max
        bg[..., c] = cv2.GaussianBlur(mx.astype(np.float32), (0, 0), k / 3)
    bg = cv2.resize(bg, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.maximum(bg, 1.0)

def darkness(rgb, dpi, bg=None):
    """d = 1 - min_c(rgb/bg): 0 on paper, ->1 on black ink / saturated ink. Lightly smoothed vs jpeg noise."""
    if bg is None:
        bg = ref_background(rgb, dpi)
    n = np.clip(rgb.astype(np.float32) / bg, 0, 1.2)
    d = 1 - np.minimum(n.min(-1), 1.0)
    return d, n

def lab_lch(rgb_u8):
    lab = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2Lab).astype(np.float32)
    L = lab[..., 0] * 100 / 255; a = lab[..., 1] - 128; b = lab[..., 2] - 128
    C = np.hypot(a, b); h = np.degrees(np.arctan2(b, a)) % 360
    return L, C, h
