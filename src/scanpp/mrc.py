"""Ink-mask + low-res colour layer machinery (the "E3/E3x" scheme of research/r3_encoding) and the mask codecs.

The cleaned page is white paper with ink.  A 1-bit ink mask (full / native resolution, lossless) is the sharp part;
a ~100 ppi colour layer carries the ink colour and is painted through the mask (/Mask on the colour image).
All sizes are in pixels of the page being encoded; `ppi` is that page's pixels-per-inch.
"""
from __future__ import annotations

import functools
import io
import logging
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger("scanpp")

# adaptive ink threshold on darkness d = 255 - min(R,G,B):  ink if d >= clip(FRAC * local_max_7x7@300(d), LO, HI)
ADAPT = (0.35, 20, 88)
COLOR_GAMMA = 3.0

_LUT_CACHE: dict = {}


def _odd_window(px_at_300: float, ppi: float) -> int:
    """Odd window size covering `px_at_300` pixels of a 300 ppi image, at least 3."""
    s = px_at_300 * ppi / 300.0
    return max(3, 2 * int(round((s - 1) / 2)) + 1)


def chan_min(rgb: np.ndarray) -> np.ndarray:
    """min over channels of a uint8 HxWx3 image (~15x faster than rgb.min(axis=2))."""
    r, g, b = cv2.split(rgb)
    return cv2.min(cv2.min(r, g), b)


def chan_max(rgb: np.ndarray) -> np.ndarray:
    r, g, b = cv2.split(rgb)
    return cv2.max(cv2.max(r, g), b)


def darkness(rgb: np.ndarray) -> np.ndarray:
    return 255 - chan_min(rgb)


def ink_mask(rgb: np.ndarray, ppi: float = 300.0, thr=ADAPT) -> np.ndarray:
    """Boolean ink mask at the resolution of `rgb` (ppi tells the 7x7@300 window how to scale)."""
    d = darkness(rgb)
    f, lo, hi = thr
    k = _odd_window(7, ppi)
    mx = cv2.dilate(d, np.ones((k, k), np.uint8))
    lut = np.ceil(np.clip(f * np.arange(256, dtype=np.float64), lo, hi)).astype(np.uint8)   # d integer: d >= t <=> d >= ceil(t)
    return d >= lut[mx]


def pushpull(col: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Smooth push-pull inpainting of ~valid pixels of `col` (HxWx3 float32); exact on valid pixels."""
    v = valid.astype(np.float32)
    lv = [(col * v[..., None], v)]
    while min(lv[-1][1].shape) > 2:
        c, w = lv[-1]
        lv.append((cv2.pyrDown(c), cv2.pyrDown(w)))
    fill = None
    for c, w in reversed(lv):
        est = c / np.maximum(w, 1e-6)[..., None]
        if fill is not None:
            up = cv2.resize(fill, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_LINEAR)
            a = np.clip(w * 4, 0, 1)[..., None]
            ia = 1 - a
            est *= a                          # est * a + up * (1 - a), same operations in place
            up *= ia
            est += up
        fill = est
    return np.where(valid[..., None], col, fill)


def color_layer(rgb: np.ndarray, ink: np.ndarray, ppi: float, col_ppi: float = 100.0) -> np.ndarray:
    """Low-res colour under the ink: darkness^3-weighted mean of the INK pixels per cell (normalized convolution over
    ink only); cells without ink are filled by push-pull.  Size round(W*col_ppi/ppi) (never larger than the page)."""
    h, w = rgb.shape[:2]
    col_ppi = min(col_ppi, ppi)
    cw, ch = max(1, round(w * col_ppi / ppi)), max(1, round(h * col_ppi / ppi))
    lut = (np.arange(256, dtype=np.float32) / 255.0) ** COLOR_GAMMA
    wgt = lut[darkness(rgb)] * ink
    num = cv2.resize(np.multiply(rgb, wgt[..., None], dtype=np.float32), (cw, ch), interpolation=cv2.INTER_AREA)   # (one temporary, not two)
    den = cv2.resize(wgt, (cw, ch), interpolation=cv2.INTER_AREA)
    valid = den > 1e-6
    if not valid.any():
        return np.zeros((ch, cw, 3), np.uint8)
    col = num / np.maximum(den, 1e-9)[..., None]
    if not valid.all():
        col = pushpull(col, valid)
    return np.clip(col + 0.5, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------- JPEG
def jpeg_bytes(arr: np.ndarray, quality: int, subsampling: int = 0) -> bytes:
    """Baseline sequential, Huffman-optimised JPEG.  subsampling 0 = 4:4:4 (the only one used for colour layers:
    4:2:0 changes the colour coding of thin pen strokes)."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=quality, subsampling=subsampling if arr.ndim == 3 else 0,
                              optimize=True, progressive=False)
    return buf.getvalue()


# ----------------------------------------------------------------------------- mask codecs
@functools.lru_cache(maxsize=1)
def jbig2_path() -> str | None:
    return shutil.which(os.environ.get("JBIG2", "jbig2"))


_warned: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(msg)


def _pbm(ink: np.ndarray) -> bytes:
    h, w = ink.shape
    return b"P4\n%d %d\n" % (w, h) + np.packbits(ink, axis=1).tobytes()      # 1 = black = ink


_PNM_WS = frozenset((0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x20))


def _png1(ink: np.ndarray) -> bytes:
    """1-bit PNG of the mask (ink = black).  jbig2 is fed this instead of a PBM when the first raster byte is whitespace-valued: leptonica's
    PNM reader takes leading raster bytes 0x09-0x0d / 0x20 for header whitespace and fails ("read error") on such masks.  (Always using
    PNG costs ~0.07 s per 300 ppi page, ~50 % on top of the jbig2 run itself; the streams are byte-identical.)"""
    h, w = ink.shape
    im = Image.frombytes("1", (w, h), np.packbits(~ink, axis=1).tobytes())          # PIL mode "1": 1 = white
    buf = io.BytesIO()
    im.save(buf, "PNG", compress_level=1)
    return buf.getvalue()


def jbig2_generic(ink: np.ndarray) -> bytes:
    """Embedded-form (`-p`) JBIG2 GENERIC region stream (never symbol mode); no globals needed.  Raises on failure."""
    exe = jbig2_path()
    if exe is None:
        raise FileNotFoundError("jbig2 not found")
    with tempfile.TemporaryDirectory(prefix="scanpp-jbig2-") as td:
        if ink.size and (np.packbits(ink[0, :8]) if ink.shape[1] else np.zeros(1, np.uint8))[0] in _PNM_WS:
            src, data = os.path.join(td, "m.png"), _png1(ink)
        else:
            src, data = os.path.join(td, "m.pbm"), _pbm(ink)
        with open(src, "wb") as f:
            f.write(data)
        r = subprocess.run([exe, "-p", "-d", src], capture_output=True, timeout=120)
    if r.returncode != 0 or not r.stdout:
        raise RuntimeError(f"jbig2 failed ({r.returncode}): {r.stderr[-200:]!r}")
    return r.stdout


def g4_strip(ink: np.ndarray) -> bytes:
    """Raw CCITT Group 4 data (single strip) for the mask via Pillow/libtiff, ink coded as BLACK runs (paper = white
    runs, the conventional polarity).  PDF: /K -1, BlackIs1 false (black -> 0 = painted).
    libtiff codes buffer bit 0 as white and bit 1 as black, so the ink bits are written as 1 (Pillow's mode "1" then
    describes the inverted picture; the TIFF wrapper is discarded, only the strip is used)."""
    h, w = ink.shape
    im = Image.frombytes("1", (w, h), np.packbits(ink, axis=1).tobytes())
    buf = io.BytesIO()
    im.save(buf, "TIFF", compression="group4", tiffinfo={278: h})
    raw = buf.getvalue()
    with Image.open(io.BytesIO(raw)) as t:
        offs, cnts = t.tag_v2[273], t.tag_v2[279]
        if isinstance(offs, int):
            offs, cnts = (offs,), (cnts,)
        if len(offs) != 1:
            raise RuntimeError("G4 TIFF has several strips")
    return raw[offs[0]: offs[0] + cnts[0]]


def encode_mask(ink: np.ndarray, prefer: str = "jbig2"):
    """Returns (data, codec, ImageSpec-ready dict).  JBIG2 generic when available; CCITT G4 fallback (logged once)."""
    h, w = ink.shape
    if prefer == "jbig2":
        try:
            return jbig2_generic(ink), "jbig2", dict(filter="JBIG2Decode", parms=None)
        except Exception as e:  # noqa: BLE001
            _warn_once("g4", f"JBIG2 unavailable/failed ({e}); falling back to CCITT G4 for ink masks")
    return g4_strip(ink), "g4", dict(filter="CCITTFaxDecode",
                                       parms={"K": -1, "Columns": w, "Rows": h, "BlackIs1": False})
