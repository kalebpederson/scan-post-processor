"""Page encoders.

Narrow interface: `encode_page(rgb, dpi, page_size_pt, encoder="auto") -> EncodedPage`.

An EncodedPage holds everything needed to emit one PDF page via pikepdf: image XObjects (raw, already filtered
stream bytes + dictionary entries) and the content stream drawing them in a page-local coordinate system whose
origin is the lower-left corner of the page box (pdfio translates to the real MediaBox origin).

Encoders (`--encoder`):
  auto     per-page rule: no-colour page -> E3 (mask + 100 ppi gray layer); highlighter / ink-heavy page -> E1;
           otherwise E3x (1-bit ink mask + 100 ppi q40 4:4:4 colour layer under it)
  mrc      always E3x (E3 gray layer on no-colour pages)
  jpeg     always E1: baseline JPEG, 150 ppi, q55, 4:4:4
  jpeg200  E1 at 200 ppi (max legibility)
`rgb` is the full-resolution cleaned page; encoders do their own resampling (never upsample).
`--quality 1..10` (QualitySettings, QUALITY_LADDER) sets the resolution / JPEG quality of the layers and, from level 7, makes `auto` store
every page as a full-page JPEG; level 5 = the constants below, i.e. the historical output (research/r8_quality).
To add an encoder write `(rgb, dpi, page_size_pt, jpeg_dpi=None) -> EncodedPage` and register it with
`@register_encoder("name")`.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Callable

from typing import TYPE_CHECKING

JPEG_Q_E1 = 55
MRC_COL_PPI = 100
MRC_COL_Q = 40
MRC_MASK_PPI = 300
DEFAULT_QUALITY = 5
QUALITY_MIN, QUALITY_MAX = 1, 10
CHROMA_THRESHOLD = 0.08       # per-pixel chroma (max-min of channels / 255) that counts as "colour"
CHROMA_FRACTION = 0.0005      # a page is gray when fewer than this fraction of pixels are coloured
DEFAULT_ENCODER = "auto"
JPEG_PPI = {"jpeg": 150, "jpeg200": 200}


@dataclass(frozen=True)
class QualitySettings:
    """The encoding knobs one `--quality` level stands for (see `quality_settings` and README "Quality").  The defaults are
    the level-5 values, i.e. the encoders' historical constants.  Every JPEG is baseline, 4:4:4; masks are JBIG2 generic / G4."""
    level: int = DEFAULT_QUALITY
    col_ppi: int = MRC_COL_PPI     # ppi of the colour / gray layer under the ink mask (E3x / E3)
    col_q: int = MRC_COL_Q         # its JPEG quality
    mask_ppi: int = MRC_MASK_PPI   # ink mask resolution (never above the page's native ppi)
    e1_ppi: int = 150              # ppi of full-page JPEG pages (E1)
    e1_q: int = JPEG_Q_E1          # their JPEG quality
    full_page: bool = False        # auto: every non-blank page is a full-page JPEG (a 1-bit mask loses stroke anti-aliasing)


def quality_settings(level) -> QualitySettings:
    """Settings for `level` (1..10), or pass a QualitySettings through (used by the study scripts)."""
    if isinstance(level, QualitySettings):
        return level
    if isinstance(level, int) and not isinstance(level, bool) and level in QUALITY_LADDER:
        return QUALITY_LADDER[level]
    raise ValueError(f"quality must be an integer {QUALITY_MIN}..{QUALITY_MAX}, got {level!r}")


# The ladder, from research/r8_quality (measured on 58 real 300 ppi pages): (level, colour-layer ppi, colour-layer q, E1 ppi, E1 q, full_page).
# Levels 1-5 keep the 1-bit ink mask + low-resolution colour layer (E3/E3x) and only turn the layer's resolution / JPEG quality (and the
# E1 quality of highlighter pages); levels 6-10 add resolution and quality, and from 7 on every page is a full-page JPEG, because a 1-bit
# mask cannot carry stroke anti-aliasing or faint pencil.  All JPEGs baseline 4:4:4; masks JBIG2 generic (or G4); mask ppi min(300, native).
_LADDER = [
    (1, 75, 25, 150, 35, False),
    (2, 75, 40, 150, 40, False),
    (3, 100, 25, 150, 45, False),
    (4, 100, 32, 150, 50, False),
    (5, 100, 40, 150, 55, False),        # = the historical defaults
    (6, 150, 40, 150, 58, False),
    (7, 150, 55, 200, 58, True),
    (8, 200, 55, 300, 58, True),
    (9, 200, 70, 300, 75, True),
    (10, 300, 85, 300, 92, True),
]
QUALITY_LADDER: dict[int, QualitySettings] = {
    n: QualitySettings(n, cp, cq, MRC_MASK_PPI, ep, eq, fp) for n, cp, cq, ep, eq, fp in _LADDER}


@dataclass
class ImageSpec:
    """One image XObject: `data` is the stream as stored (already compressed with `filter`)."""
    data: bytes
    width: int
    height: int
    colorspace: str | None = "DeviceRGB"   # PDF name without the slash (None for an /ImageMask)
    bpc: int = 8
    filter: str | None = "DCTDecode"       # PDF name without the slash, or None for raw
    extra: dict = field(default_factory=dict)   # additional entries, e.g. {"Decode": [1, 0], "ImageMask": True}
    smask: "ImageSpec | None" = None
    mask: "ImageSpec | None" = None        # explicit stencil mask (/Mask), an /ImageMask image


@dataclass
class EncodedPage:
    xobjects: dict[str, ImageSpec]
    content: bytes
    media_box: tuple = (0.0, 0.0, 0.0, 0.0)   # filled in by the pipeline from the source page
    rotate: int = 0
    info: dict = field(default_factory=dict)  # stats for the report (scheme, gray, bytes, ...)
    ocr: object = None                        # ocr.OcrLayer (invisible text) or None

    @property
    def nbytes(self) -> int:
        def n(s: ImageSpec) -> int:
            return len(s.data) + sum(n(x) for x in (s.smask, s.mask) if x)
        return sum(n(x) for x in self.xobjects.values())


if TYPE_CHECKING:
    import numpy as np

Encoder = Callable[..., EncodedPage]
_ENCODERS: dict[str, Encoder] = {}


def register_encoder(name: str):
    def deco(fn: Encoder) -> Encoder:
        _ENCODERS[name] = fn
        return fn
    return deco


def encoder_names() -> list[str]:
    return sorted(_ENCODERS)


def encode_page(rgb: np.ndarray, dpi: float, page_size_pt: tuple, encoder: str = DEFAULT_ENCODER,
                jpeg_dpi: int | None = None, quality: int | QualitySettings = DEFAULT_QUALITY) -> EncodedPage:
    """`rgb`: cleaned page at its native resolution `dpi`.  `jpeg_dpi` (the --dpi option) overrides the ppi of the
    jpeg/jpeg200 encoders only.  `quality`: the --quality level 1..10 (encoding only; 5 = the historical settings)."""
    try:
        fn = _ENCODERS[encoder]
    except KeyError:
        raise ValueError(f"unknown encoder {encoder!r}; available: {', '.join(encoder_names())}") from None
    return fn(rgb, dpi, page_size_pt, jpeg_dpi, quality_settings(quality))


def has_color(rgb: np.ndarray, thr: float = CHROMA_THRESHOLD, frac: float = CHROMA_FRACTION) -> bool:
    """True when at least `frac` of the pixels have chroma (max-min over channels, /255) above `thr`."""
    import numpy as np
    if rgb.ndim == 2:
        return False
    from .mrc import chan_max, chan_min
    chroma = chan_max(rgb).astype(np.int16) - chan_min(rgb)
    return float(np.count_nonzero(chroma > thr * 255)) / chroma.size >= frac


def _full_page_content(w_pt: float, h_pt: float, name: str = "Im0") -> bytes:
    return f"q\n{w_pt:.4f} 0 0 {h_pt:.4f} 0 0 cm\n/{name} Do\nQ\n".encode("ascii")


def _blank_page(dpi: float) -> EncodedPage:
    """Pure white page: empty content stream at the page's MediaBox."""
    return EncodedPage({}, b"", info={"scheme": "blank", "dpi": round(float(dpi), 2)})


def _is_white(rgb: np.ndarray) -> bool:
    import numpy as np
    from .mrc import chan_min
    mn = chan_min(rgb)
    return float(np.count_nonzero(mn < 255)) / mn.size < 2e-4


# ----------------------------------------------------------------------------- E1: plain JPEG
def _encode_e1(rgb: np.ndarray, dpi: float, page_size_pt: tuple, ppi: int, scheme: str, q: int = JPEG_Q_E1) -> EncodedPage:
    import cv2
    from .mrc import jpeg_bytes
    gray = not has_color(rgb)
    out_ppi = float(dpi)
    if ppi < dpi * 0.98:                                  # never upsample
        s = ppi / dpi
        h, w = rgb.shape[:2]
        rgb = cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        out_ppi = float(ppi)
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if gray else rgb
    h, w = img.shape[:2]
    spec = ImageSpec(jpeg_bytes(img, q, 0), w, h, "DeviceGray" if gray else "DeviceRGB", 8, "DCTDecode")
    return EncodedPage({"Im0": spec}, _full_page_content(*page_size_pt),
                       info={"scheme": scheme, "gray": gray, "dpi": round(out_ppi, 2), "jpeg_ppi": round(out_ppi, 2), "jpeg_q": q})


def _jpeg_encoder(name: str):
    def fn(rgb, dpi, page_size_pt, jpeg_dpi=None, qs=QualitySettings()):
        if _is_white(rgb):
            return _blank_page(dpi)
        # ppi: --dpi if given, else the encoder's own (150 / 200), raised to the quality level's ppi when that is higher
        return _encode_e1(rgb, dpi, page_size_pt, jpeg_dpi or max(JPEG_PPI[name], qs.e1_ppi), "e1", qs.e1_q)
    register_encoder(name)(fn)
    return fn


encode_jpeg = _jpeg_encoder("jpeg")
encode_jpeg200 = _jpeg_encoder("jpeg200")


# ----------------------------------------------------------------------------- E3 / E3x: ink mask + colour layer
def _to_mask_ppi(rgb: np.ndarray, dpi: float, mask_ppi: int = MRC_MASK_PPI):
    """Masks are made at min(mask_ppi, native) ppi: oversampled inputs are area-downsampled, others stay native."""
    if dpi > mask_ppi * 1.02:
        import cv2
        s = mask_ppi / dpi
        h, w = rgb.shape[:2]
        return cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA), float(mask_ppi)
    return rgb, float(dpi)


def _encode_mrc(rgb: np.ndarray, dpi: float, page_size_pt: tuple, gray: bool, qs: QualitySettings = QualitySettings()) -> EncodedPage:
    import cv2
    from . import mrc
    rgb, ppi = _to_mask_ppi(rgb, dpi, qs.mask_ppi)
    ink = mrc.ink_mask(rgb, ppi)
    if not ink.any():
        return _blank_page(dpi)
    h, w = ink.shape
    col = mrc.color_layer(rgb, ink, ppi, qs.col_ppi)
    if gray:
        col = cv2.cvtColor(col, cv2.COLOR_RGB2GRAY)
    ch, cw = col.shape[:2]
    data, codec, kw = mrc.encode_mask(ink)
    extra = {"ImageMask": True}
    if kw["parms"]:
        extra["DecodeParms"] = kw["parms"]
    mask = ImageSpec(data, w, h, None, 1, kw["filter"], extra)
    fg = ImageSpec(mrc.jpeg_bytes(col, qs.col_q, 0), cw, ch, "DeviceGray" if gray else "DeviceRGB", 8, "DCTDecode",
                   mask=mask)
    return EncodedPage({"Fg": fg}, _full_page_content(*page_size_pt, name="Fg"),
                       info={"scheme": "e3" if gray else "e3x", "gray": gray, "dpi": round(ppi, 2),
                             "mask_ppi": round(ppi, 2), "col_ppi": round(ppi * cw / w, 1), "mask_codec": codec,
                             "mask_bytes": len(data), "color_bytes": len(fg.data)})


@register_encoder("mrc")
def encode_mrc(rgb, dpi, page_size_pt, jpeg_dpi=None, qs=QualitySettings()):
    return _encode_mrc(rgb, dpi, page_size_pt, not _page_has_ink_color(rgb, dpi), qs)


# ----------------------------------------------------------------------------- per-page selection
def _scaled_odd(px_at_300: float, dpi: float) -> int:
    return max(3, int(round(px_at_300 * dpi / 300.0)) | 1)


def has_highlighter(rgb: np.ndarray, dpi: float = 300.0) -> bool:
    """Pale saturated region (>= 5000 px at 300 ppi) surviving a 25 px opening: highlighter."""
    import cv2
    import numpy as np
    s = dpi / 300.0
    from .mrc import chan_max, chan_min
    mn = chan_min(rgb)
    mx = chan_max(rgb)
    pale = ((mx.astype(np.int16) - mn) >= 25) & (mn >= 100) & (mn < 250)
    k = _scaled_odd(25, dpi)
    reg = cv2.morphologyEx(pale.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    k = _scaled_odd(13, dpi)
    reg = cv2.morphologyEx(reg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return int(np.count_nonzero(reg)) >= 5000 * s * s


def _color_stats(rgb: np.ndarray, dpi: float):
    """(ink pixel count, saturated ink pixel count) with ink = min(rgb) < 191, saturated = chroma > 40."""
    import numpy as np
    from .mrc import chan_max, chan_min
    mn = chan_min(rgb)
    ink = mn < 191
    sat = ink & ((chan_max(rgb).astype(np.int16) - mn) > 40)
    return int(np.count_nonzero(ink)), int(np.count_nonzero(sat))


def _page_has_ink_color(rgb: np.ndarray, dpi: float) -> bool:
    s2 = (dpi / 300.0) ** 2
    n_ink, n_col = _color_stats(rgb, dpi)
    return not (n_col < 1500 * s2 or n_col / max(1, n_ink) < 0.005)


def select_scheme(rgb: np.ndarray, dpi: float) -> str:
    """Round-2 rule: 'blank' | 'e3' (no colour: mask + gray layer) | 'e1' (highlighter / ink-heavy art) | 'e3x'."""
    import numpy as np
    from .mrc import chan_min
    mn = chan_min(rgb)
    nonwhite = float(np.count_nonzero(mn < 255)) / mn.size
    if nonwhite < 2e-4:
        return "blank"
    n_ink, n_col = _color_stats(rgb, dpi)
    s2 = (dpi / 300.0) ** 2
    if n_col < 1500 * s2 or n_col / max(1, n_ink) < 0.005:
        return "e3"
    if n_ink / mn.size > 0.15 or nonwhite > 0.30 or has_highlighter(rgb, dpi):
        return "e1"
    return "e3x"


@register_encoder("auto")
def encode_auto(rgb, dpi, page_size_pt, jpeg_dpi=None, qs=QualitySettings()):
    scheme = select_scheme(rgb, dpi)
    if scheme == "blank":
        return _blank_page(dpi)
    if scheme == "e1" or qs.full_page:
        return _encode_e1(rgb, dpi, page_size_pt, qs.e1_ppi, "e1", qs.e1_q)
    return _encode_mrc(rgb, dpi, page_size_pt, scheme == "e3", qs)
