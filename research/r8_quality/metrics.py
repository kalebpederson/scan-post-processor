"""Fidelity metrics of a rendered encoded page against the cleaned full-resolution page (both uint8 RGB, same grid)."""
from __future__ import annotations

import subprocess

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000
from skimage.metrics import structural_similarity


def render_pdf(path, ppi: int = 300) -> np.ndarray:
    """poppler render of page 1 at `ppi`, RGB uint8."""
    r = subprocess.run(["pdftoppm", "-r", str(ppi), "-f", "1", "-l", "1", str(path)], capture_output=True, check=True)
    arr = cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)


def _fit(ref: np.ndarray, ren: np.ndarray) -> np.ndarray:
    h, w = ref.shape[:2]
    out = np.full_like(ref, 255)
    hh, ww = min(h, ren.shape[0]), min(w, ren.shape[1])
    out[:hh, :ww] = ren[:hh, :ww]
    return out


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _grad(g: np.ndarray) -> np.ndarray:
    g = g.astype(np.float32)
    return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))


def metrics(ref: np.ndarray, ren: np.ndarray, rng: np.random.Generator | None = None) -> dict:
    ren = _fit(ref, ren)
    gr = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    gn = cv2.cvtColor(ren, cv2.COLOR_RGB2GRAY)
    out: dict = {}
    out["ssim"] = float(structural_similarity(gr, gn, data_range=255))
    mse = float(np.mean((gr.astype(np.float32) - gn.astype(np.float32)) ** 2))
    out["psnr"] = float(10 * np.log10(255.0 ** 2 / max(mse, 1e-9)))
    # colour identity: CIEDE2000 over ink pixels (ref min-channel < 150), and over coloured ink (also chroma C* > 25)
    mn = ref.min(axis=2)
    lab_r = _lab(ref)
    chroma = np.hypot(lab_r[..., 1], lab_r[..., 2])
    rng = rng or np.random.default_rng(0)
    for name, m in (("de_ink", mn < 150), ("de_col", (mn < 200) & (chroma > 25))):
        idx = np.flatnonzero(m.ravel())
        if idx.size == 0:
            out[name] = None
            out[name + "_n"] = 0
            continue
        if idx.size > 150000:
            idx = rng.choice(idx, 150000, replace=False)
        a = lab_r.reshape(-1, 3)[idx][None]
        b = _lab(ren).reshape(-1, 3)[idx][None]
        out[name] = float(deltaE_ciede2000(a, b).mean())
        out[name + "_n"] = int(m.sum())
    # stroke edge sharpness: gradient energy at the reference's strong edges, render / reference (1 = as sharp)
    gref, gren = _grad(gr), _grad(gn)
    em = gref > 80
    out["edge_sharp"] = float(gren[em].sum() / max(gref[em].sum(), 1e-9)) if em.any() else None
    # ink loss: reference ink (solid min<128, faint 128..199) rendered white (>=245); speckle: reference-white rendered < 245
    mnn = ren.min(axis=2)
    solid, faint = mn < 128, (mn >= 128) & (mn < 200)
    out["loss_solid"] = float((mnn[solid] >= 245).mean()) if solid.any() else None
    out["loss_faint"] = float((mnn[faint] >= 245).mean()) if faint.any() else None
    out["speckle"] = float((mnn[mn == 255] < 245).mean())
    return out
