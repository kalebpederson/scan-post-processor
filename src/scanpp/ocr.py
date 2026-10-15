"""Optional invisible, searchable text layer (--ocr) from tesseract.

Per page, tesseract runs on the cleaned, deskewed, upright page at full resolution and writes (one recognition pass)
a text-only PDF (``textonly_pdf=1``: text render mode 3 in tesseract's glyphless font, positioned per word) and a TSV
with per-word confidences.  Words below ``min_conf`` are removed from the text-only content stream (the TJ operator of
a word is dropped; the relative positioning operators stay, so the other words do not move).  The result is stored as
a Form XObject that is drawn after the page image with the matrix that maps tesseract's page (upright, in points at the
OCR dpi) onto the real page: scale to the MediaBox, undo /Rotate, translate by the MediaBox origin.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import pikepdf
from pikepdf import Name

log = logging.getLogger("scanpp")

MIN_CONF = 60.0           # words below this tesseract confidence (0-100) are not written (see README for the data)
TIMEOUT_S = 300
MARKER = "/ScanppOCR"     # key in the text Form XObject: "this page carries a scanpp text layer"
FORM_NAME = "ScanppOCR"


@dataclass
class OcrLayer:
    pdf: bytes                                   # filtered text-only PDF (1 page) from tesseract
    words: int = 0
    info: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- availability
_avail_cache: dict = {}


def tesseract_languages() -> set[str] | None:
    """Installed tesseract languages, or None if tesseract is missing/unusable (cached per PATH)."""
    key = os.environ.get("PATH", "")
    if key not in _avail_cache:
        exe = shutil.which("tesseract")
        langs = None
        if exe:
            try:
                p = subprocess.run([exe, "--list-langs"], capture_output=True, text=True, timeout=30)
                if p.returncode == 0:
                    langs = {s.strip() for s in p.stdout.splitlines()[1:] if s.strip()}
            except (OSError, subprocess.SubprocessError):
                langs = None
        _avail_cache[key] = langs
    return _avail_cache[key]


def check_available(lang: str) -> str | None:
    """None if OCR can run, else the reason (logged once by the caller)."""
    langs = tesseract_languages()
    if langs is None:
        return "tesseract is not installed (or not working)"
    missing = [l for l in lang.split("+") if l and l not in langs]
    if missing:
        return f"tesseract language data missing: {', '.join(missing)} (installed: {', '.join(sorted(langs))})"
    return None


# ----------------------------------------------------------------------------- recognition
def _decode_word(s: bytes) -> str:
    try:
        return s.decode("utf-16-be").strip()
    except UnicodeDecodeError:
        return ""


def _parse_tsv(text: str) -> list[tuple[str, float]]:
    out = []
    for line in text.splitlines()[1:]:
        c = line.split("\t")
        if len(c) >= 12 and c[0] == "5":
            w = c[11].strip()
            if w:
                out.append((w, float(c[10])))
    return out


def _filter_pdf(pdf_bytes: bytes, words: list[tuple[str, float]], min_conf: float):
    """Drop low-confidence words from the text-only PDF. Returns (new_pdf_bytes or None, kept, total)."""
    import io
    pdf = pikepdf.open(io.BytesIO(pdf_bytes))
    page = pdf.pages[0]
    ops = list(pikepdf.parse_content_stream(page))
    tjs = [i for i, ins in enumerate(ops) if str(ins.operator) == "TJ"]
    texts = [_decode_word(bytes(ins.operands[0][0])) if len(ins.operands[0]) else "" for ins in (ops[i] for i in tjs)]
    if len(tjs) == len(words) and all(t == w for t, (w, _) in zip(texts, words)):
        drop = {i for i, (_, c) in zip(tjs, words) if c < min_conf}
    elif len(tjs) == 0:
        return None, 0, 0
    else:                                        # layout did not map 1:1 to the TSV: keep everything (never mis-filter)
        log.debug("ocr: text-only PDF words do not match TSV (%d vs %d); confidence filter skipped", len(tjs), len(words))
        drop = set()
    kept = len(tjs) - len(drop)
    if kept == 0:
        return None, 0, len(tjs)
    new_ops = [ins for i, ins in enumerate(ops) if i not in drop and str(ins.operator) not in ("q", "Q", "cm")]
    page.Contents = pdf.make_stream(pikepdf.unparse_content_stream(new_ops))
    buf = io.BytesIO()
    pdf.save(buf)
    return buf.getvalue(), kept, len(tjs)


def ocr_page(gray: np.ndarray, dpi: float, lang: str = "eng", min_conf: float = MIN_CONF) -> tuple[OcrLayer | None, dict]:
    """OCR an upright 8-bit gray page. Never raises: returns (layer or None, stats)."""
    stats: dict = {"lang": lang, "min_conf": min_conf, "words": 0, "words_raw": 0}
    t_wall = time.perf_counter()
    c0 = os.times()
    try:
        exe = shutil.which("tesseract")
        if exe is None:
            raise RuntimeError("tesseract not found")
        env = dict(os.environ, OMP_THREAD_LIMIT="1")
        with tempfile.TemporaryDirectory(prefix="scanpp-ocr-") as d:
            inp = os.path.join(d, "in.pgm")
            if not cv2.imwrite(inp, gray):
                raise RuntimeError("could not write OCR input")
            cmd = [exe, inp, os.path.join(d, "out"), "-l", lang, "--dpi", str(max(70, int(round(dpi)))),
                   "-c", "textonly_pdf=1", "pdf", "tsv"]
            p = subprocess.run(cmd, capture_output=True, env=env, timeout=TIMEOUT_S, stdin=subprocess.DEVNULL)
            if p.returncode != 0:
                raise RuntimeError(f"tesseract exited {p.returncode}: {p.stderr.decode(errors='replace').strip()[:300]}")
            pdf_bytes = open(os.path.join(d, "out.pdf"), "rb").read()
            words = _parse_tsv(open(os.path.join(d, "out.tsv"), encoding="utf-8", errors="replace").read())
        stats["words_raw"] = len(words)
        if words:
            stats["conf_mean"] = round(float(np.mean([c for _, c in words])), 1)
        data, kept, total = _filter_pdf(pdf_bytes, words, min_conf)
        stats["words"] = kept
        stats["words_dropped_low_conf"] = total - kept
        if data is None:
            return None, stats
        return OcrLayer(data, kept), stats
    except Exception as e:  # noqa: BLE001  - a failed page just gets no text layer
        stats["error"] = f"{type(e).__name__}: {e}"
        log.warning("OCR failed for a page (emitting it without text): %s", stats["error"])
        return None, stats
    finally:
        c1 = os.times()
        stats["wall"] = round(time.perf_counter() - t_wall, 3)
        stats["cpu"] = round((c1.children_user + c1.children_system) - (c0.children_user + c0.children_system), 3)


# ----------------------------------------------------------------------------- embedding
def _rot_matrix(rotate: int, w: float, h: float) -> pikepdf.Matrix:
    """Maps user space (w x h box) to the displayed (rotated clockwise by /Rotate) page."""
    return {0: pikepdf.Matrix(1, 0, 0, 1, 0, 0),
            90: pikepdf.Matrix(0, -1, 1, 0, 0, w),
            180: pikepdf.Matrix(-1, 0, 0, -1, w, h),
            270: pikepdf.Matrix(0, 1, -1, 0, h, 0)}[rotate % 360]


def layer_matrix(tess_size: tuple, media_box: tuple, rotate: int) -> pikepdf.Matrix:
    """Matrix from tesseract page space (upright display, origin bottom-left) to the page's user space."""
    w, h = media_box[2] - media_box[0], media_box[3] - media_box[1]
    dw, dh = (h, w) if rotate % 180 else (w, h)          # displayed (upright) size
    scale = pikepdf.Matrix(dw / tess_size[0], 0, 0, dh / tess_size[1], 0, 0)
    inv = _rot_matrix(rotate, w, h).inverse()
    return scale @ inv


def _font_key(font) -> str:
    h = hashlib.sha1()
    d = font.DescendantFonts[0]
    for obj in (d.FontDescriptor.FontFile2, d.get("/CIDToGIDMap"), font.get("/ToUnicode")):
        if isinstance(obj, pikepdf.Stream):
            h.update(obj.read_bytes())
    return h.hexdigest()


def embed_layer(pdf: pikepdf.Pdf, layer: OcrLayer, media_box: tuple, rotate: int, font_cache: dict):
    """Add the text layer to `pdf`. Returns (form XObject, content bytes to append to the page's content stream);
    the form must be registered in the page's /XObject dict under FORM_NAME. The glyphless font is shared by all
    pages of the document."""
    import io
    src = pikepdf.open(io.BytesIO(layer.pdf))
    sp = src.pages[0]
    tw, th = (float(v) for v in sp.MediaBox[2:])
    font = next(iter(sp.Resources.Font.values()))
    key = _font_key(font)
    if key not in font_cache:
        font_cache[key] = pdf.copy_foreign(font)
    fname = next(iter(sp.Resources.Font.keys()))[1:]
    form = pikepdf.Stream(pdf, sp.Contents.read_bytes())
    form.Type = Name.XObject
    form.Subtype = Name.Form
    form.BBox = pikepdf.Array([0, 0, tw, th])
    form.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary({"/" + fname: font_cache[key]}))
    form[MARKER] = True
    m = layer_matrix((tw, th), media_box, rotate)
    a, b, c, d, e, f = m.shorthand
    ops = f"q {a:.6f} {b:.6f} {c:.6f} {d:.6f} {e:.4f} {f:.4f} cm /{FORM_NAME} Do Q\n".encode()
    return form, ops
