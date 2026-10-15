"""Single-image inputs (JPEG / PNG / WebP): read to RGB, write back in the same format."""
from __future__ import annotations

import io
import logging
from pathlib import Path


log = logging.getLogger("scanpp")

SUPPORTED = {"JPEG", "PNG", "WEBP"}
DEFAULT_DPI = 300.0
JPEG_QUALITY = 85


def sniff_format(path) -> str | None:
    """Return 'PDF', one of SUPPORTED, or None (by magic bytes, not by extension or MIME)."""
    with open(path, "rb") as f:
        head = f.read(16)
    if head.lstrip()[:5] == b"%PDF-" or head[:5] == b"%PDF-":
        return "PDF"
    if head[:3] == b"\xff\xd8\xff":
        return "JPEG"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "WEBP"
    return None


def _to_rgb_u8(im):
    """Any Pillow image -> uint8 HxWx3 RGB.  Transparency is composited onto white (not dropped: dropping alpha would
    expose whatever RGB values sit under transparent pixels); palette, LA, PA, premultiplied, CMYK, 1-bit and 16/32-bit
    integer / float grayscale are all handled (high-bit-depth data is scaled, never clipped)."""
    import numpy as np
    from PIL import Image
    mode = im.mode
    if mode in ("I;16", "I;16L", "I;16B", "I;16N", "I", "F"):
        a = np.asarray(im).astype(np.float64)
        if mode == "F":
            hi = 1.0 if a.size and a.max() <= 1.0 else 255.0
        elif mode == "I" and a.size and a.max() <= 255:
            hi = 255.0                                        # 8-bit values stored in a 32-bit container
        else:
            hi = 65535.0
        g = np.clip(a / hi * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(np.repeat(g[..., None], 3, axis=2))
    has_alpha = (mode in ("RGBA", "LA", "PA", "RGBa", "La")
                 or (mode in ("P", "L", "RGB", "1") and "transparency" in im.info))
    if has_alpha:
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return np.ascontiguousarray(np.asarray(Image.alpha_composite(bg, rgba).convert("RGB")))
    return np.ascontiguousarray(np.asarray(im.convert("RGB")))


def read_image(path):
    """Returns (rgb uint8, dpi, format). dpi falls back to 300 when absent or implausible."""
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        fmt = im.format
        dpi = im.info.get("dpi")
        im = ImageOps.exif_transpose(im)
        rgb = _to_rgb_u8(im)
    try:
        d = float(dpi[0]) if dpi and dpi[0] else 0.0
    except (TypeError, ValueError):
        d = 0.0
    if not (50 <= d <= 2400):
        log.info("%s: no usable dpi metadata (%s), assuming %g", path, dpi, DEFAULT_DPI)
        d = DEFAULT_DPI
    return rgb, d, fmt


def encode_image(rgb, dpi: float, fmt: str) -> bytes:
    import numpy as np
    from PIL import Image
    from .encode import has_color
    gray = not has_color(rgb)
    buf = io.BytesIO()
    if fmt == "JPEG":
        im = Image.fromarray(rgb)
        if gray:
            im = im.convert("L")
        im.save(buf, "JPEG", quality=JPEG_QUALITY, subsampling=2, optimize=True, dpi=(dpi, dpi))
    elif fmt == "PNG":
        if gray:
            im = Image.fromarray(rgb).convert("L")
        else:
            im = Image.fromarray(rgb)
            sample = np.unique(rgb[::7, ::7].reshape(-1, 3), axis=0)
            if len(sample) <= 256:   # cheap pre-check before the exact palette test
                flat = rgb.reshape(-1, 3)
                packed = (flat[:, 0].astype(np.uint32) << 16) | (flat[:, 1].astype(np.uint32) << 8) | flat[:, 2]
                uniq = np.unique(packed)
                if len(uniq) <= 256:
                    idx = np.searchsorted(uniq, packed).astype(np.uint8).reshape(rgb.shape[:2])
                    im = Image.fromarray(idx, "P")
                    pal = []
                    for c in uniq:
                        pal += [(int(c) >> 16) & 255, (int(c) >> 8) & 255, int(c) & 255]
                    im.putpalette(pal)
        im.save(buf, "PNG", optimize=True, dpi=(dpi, dpi))
    elif fmt == "WEBP":
        Image.fromarray(rgb).save(buf, "WEBP", lossless=True, quality=100, method=4)  # WebP has no dpi field
    else:
        raise ValueError(f"unsupported image format {fmt}")
    return buf.getvalue()


def validate_image(path, size: tuple, fmt: str) -> None:
    from PIL import Image
    with Image.open(path) as im:
        im.load()
        if im.format != fmt:
            raise ValueError(f"output format {im.format}, expected {fmt}")
        if im.size != size:
            raise ValueError(f"output size {im.size}, expected {size}")
