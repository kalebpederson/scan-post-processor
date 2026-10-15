"""Synthetic halftoned light text for the cleaner tests / bench: render gray text, screen it with a rotated dot screen, scan-blur it."""
import cv2
import numpy as np


def render_text(shape=(700, 1500), dpi=300, text=("Go to numbered bookmark", "Show tools rails"), scale=1.0, thickness=2):
    """Binary glyph coverage (float 0..1, anti-aliased) of a few text lines on white (about 8-9 pt at 300 dpi for scale 1.0)."""
    img = np.zeros(shape, np.uint8)
    y = int(0.35 * dpi)
    for i, t in enumerate(text):
        cv2.putText(img, t, (int(0.2 * dpi), y + int(i * 0.55 * dpi)), cv2.FONT_HERSHEY_DUPLEX, scale * dpi / 150, 255, thickness, cv2.LINE_AA)
    return img.astype(np.float32) / 255.0


def halftone(darkness, lpi=150.0, dpi=300, angle_deg=45.0, dot_value=40, blur_px=0.7, noise=2.0, seed=0):
    """Screen a darkness map (0..1) with a cosine dot screen; dots are `dot_value` gray on white; then scanner blur + noise."""
    h, w = darkness.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    p = dpi / lpi
    a = np.deg2rad(angle_deg)
    u = (xx * np.cos(a) + yy * np.sin(a)) / p
    v = (-xx * np.sin(a) + yy * np.cos(a)) / p
    thr = 0.5 + 0.25 * (np.cos(2 * np.pi * u) + np.cos(2 * np.pi * v))      # 0..1 threshold matrix
    dot = (darkness > thr).astype(np.float32)
    g = 255.0 - dot * (255.0 - dot_value)
    g = cv2.GaussianBlur(g, (0, 0), blur_px)
    g += np.random.default_rng(seed).normal(0, noise, g.shape).astype(np.float32)
    return np.clip(g, 0, 255).astype(np.uint8)


def halftone_page(tone=0.30, lpi=150.0, dpi=300, **kw):
    """Returns (rgb uint8 page, glyph coverage map). `tone` = darkness of the printed gray (0.30 = light gray text)."""
    cov = render_text(dpi=dpi, **kw)
    g = halftone(cov * tone, lpi=lpi, dpi=dpi)
    return np.repeat(g[..., None], 3, 2), cov
