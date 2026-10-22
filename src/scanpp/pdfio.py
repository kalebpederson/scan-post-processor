"""PDF reading (scan classification, lossless page-image extraction) and writing (EncodedPage -> PDF)."""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pikepdf
from pikepdf import Name
from pikepdf.models import PdfImage

from . import __version__, ocr
from .encode import EncodedPage, ImageSpec

log = logging.getLogger("scanpp")

_ALLOWED_OPS = {"q", "Q", "cm", "Do"}

MARK_DOC = "/ScanppVersion"      # document-info key carried by every file scanpp writes
MARK_PAGE = "/ScanppPage"        # page-dictionary key carried by every page scanpp writes


@dataclass
class PageInput:
    """What a worker needs to process one page: compressed image bytes (or raw pixels as fallback) + geometry."""
    index: int
    kind: str                  # "jpeg": data is a JPEG file; "raw": data is packed HxWx3 uint8 RGB
    data: bytes
    size: tuple                # (width_px, height_px)
    media_box: tuple           # (x0, y0, x1, y1) in pt
    rotate: int                # 0/90/180/270
    dpi: float                 # native resolution from pixel size vs MediaBox


# ----------------------------------------------------------------------------- page helpers
def _inherited(page, key):
    node = page.obj
    while node is not None:
        if key in node:
            return node[key]
        node = node.get("/Parent")
    return None


def page_media_box(page) -> tuple:
    mb = _inherited(page, "/MediaBox")
    if mb is None:
        return (0.0, 0.0, 612.0, 792.0)
    x0, y0, x1, y1 = (float(v) for v in mb)
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def page_rotate(page) -> int:
    r = _inherited(page, "/Rotate")
    return int(r) % 360 if r is not None else 0


def detect_dpi(width_px: int, height_px: int, media_box: tuple) -> float:
    """Native resolution of an image that fills `media_box` (points)."""
    bw, bh = media_box[2] - media_box[0], media_box[3] - media_box[1]
    if bw <= 0 or bh <= 0:
        raise ValueError(f"degenerate MediaBox {media_box}")
    return (width_px / (bw / 72.0) + height_px / (bh / 72.0)) / 2.0


# ----------------------------------------------------------------------------- classification
def has_scanpp_text_layer(page) -> bool:
    xo = page.resources.get("/XObject")
    if xo is None:
        return False
    return any(v.get("/Subtype") == Name.Form and ocr.MARKER in v for v in xo.values())


def scanpp_marker(pdf: pikepdf.Pdf) -> str | None:
    """Version string if `pdf` was written by scanpp (explicit document or page marker), else None."""
    try:
        info = pdf.docinfo
        if info is not None and MARK_DOC in info:
            return str(info[MARK_DOC])
    except Exception:  # noqa: BLE001
        pass
    for page in pdf.pages:
        if MARK_PAGE in page.obj:
            return str(page.obj[MARK_PAGE])
    return None


def _smask_effectively_opaque(sm, border: int = 2, frac: float = 0.995) -> bool:
    """Legacy ImageMagick output carries a soft mask that is all 255 (or 255 except a resize border). scanpp itself never writes
    an /SMask, so such a mask is ignored. True only if >= `frac` of the pixels are 255 and every other pixel is within `border` px
    of the edge. Any failure to decode counts as not opaque."""
    try:
        if not isinstance(sm, pikepdf.Stream) or sm.get("/Subtype") != Name.Image or sm.get("/ImageMask", False) or "/Mask" in sm:
            return False
        if "/SMask" in sm or "/Matte" in sm:
            return False
        a = np.asarray(PdfImage(sm).as_pil_image().convert("L"))
        if a.ndim != 2 or a.size == 0:
            return False
        low = a < 255
        n_low = int(low.sum())
        if n_low == 0:
            return True
        if n_low > (1.0 - frac) * a.size:
            return False
        inner = low[border:-border, border:-border] if min(a.shape) > 2 * border else low[:0, :0]
        return not inner.any()
    except Exception:  # noqa: BLE001
        return False


def classify_page(page) -> tuple[bool, str]:
    """Is this page a plain scan: one image XObject drawn by q/cm/Do/Q only, no fonts or text?
    (Pages that carry scanpp's own invisible text layer are scanpp output, not plain scans.)"""
    if has_scanpp_text_layer(page):
        return False, "already processed by scanpp (has an OCR text layer)"
    res = page.resources
    if "/Font" in res and len(res.Font) > 0:
        return False, "has fonts"
    xobjs = res.get("/XObject")
    if xobjs is None or len(xobjs) != 1:
        return False, f"{0 if xobjs is None else len(xobjs)} XObjects (need exactly 1)"
    img = next(iter(xobjs.values()))
    if img.get("/Subtype") != Name.Image:
        return False, "XObject is not an image"
    if "/Mask" in img or "/SMaskInData" in img or ("/SMask" in img and not _smask_effectively_opaque(img.SMask)):
        return False, "image has a mask (already MRC-encoded?)"
    try:
        ops = pikepdf.parse_content_stream(page)
    except Exception as e:  # noqa: BLE001
        return False, f"unparseable content stream ({e})"
    n_do = 0
    for ins in ops:
        if isinstance(ins, pikepdf.ContentStreamInlineImage):
            return False, "inline image"
        op = str(ins.operator)
        if op not in _ALLOWED_OPS:
            return False, f"operator {op!r} (text/vector content)"
        if op == "Do":
            n_do += 1
        if op == "cm":
            a, b, c, d = (float(v) for v in ins.operands[:4])
            if abs(b) > 1e-6 or abs(c) > 1e-6 or a <= 0 or d <= 0:
                return False, "image drawn with rotation/flip transform"
    if n_do != 1:
        return False, f"{n_do} Do operators"
    return True, "ok"


def classify_pdf(pdf: pikepdf.Pdf) -> tuple[bool, str]:
    """True if every page is a plain scan page. Returns (is_scan, reason)."""
    if len(pdf.pages) == 0:
        return False, "no pages"
    if (v := scanpp_marker(pdf)) is not None:
        return False, f"already processed by scanpp {v}"
    for i, page in enumerate(pdf.pages):
        ok, why = classify_page(page)
        if not ok:
            return False, f"page {i + 1}: {why}"
    return True, "ok"


# ----------------------------------------------------------------------------- extraction
def _plain_dct(img) -> bool:
    """DCTDecode single filter, Device/ICC RGB or gray, no Decode/mask: bytes can go to Pillow as-is."""
    f = img.get("/Filter")
    if f is None or (isinstance(f, pikepdf.Array) and len(f) == 1):
        f = f[0] if f is not None else None
    if f != Name.DCTDecode:
        return False
    if "/Decode" in img or "/SMask" in img or "/Mask" in img or img.get("/ImageMask", False):
        return False
    cs = img.get("/ColorSpace")
    if cs in (Name.DeviceRGB, Name.DeviceGray):
        return True
    if isinstance(cs, pikepdf.Array) and len(cs) == 2 and cs[0] == Name.ICCBased:
        return int(cs[1].get("/N", 0)) in (1, 3)
    return False


_NON_RAW_FILTERS = {"/DCTDecode", "/JPXDecode", "/CCITTFaxDecode", "/JBIG2Decode"}
_NCOMP = {"/DeviceGray": 1, "/CalGray": 1, "/DeviceRGB": 3, "/CalRGB": 3, "/Lab": 3, "/DeviceCMYK": 4,
          "/Indexed": 1, "/Separation": 1, "/G": 1, "/RGB": 3, "/CMYK": 4, "/I": 1}


def _filters(img) -> list[str]:
    f = img.get("/Filter")
    if f is None:
        return []
    return [str(x) for x in f] if isinstance(f, pikepdf.Array) else [str(f)]


def _ncomp(img):
    """Colour components per pixel of an image XObject, or None if unknown."""
    if img.get("/ImageMask", False):
        return 1
    cs = img.get("/ColorSpace")
    if cs is None:
        return None
    if isinstance(cs, pikepdf.Array):
        if len(cs) == 0:
            return None
        if cs[0] == Name.ICCBased:
            try:
                return int(cs[1].get("/N", 0)) or None
            except Exception:  # noqa: BLE001
                return None
        return _NCOMP.get(str(cs[0]))
    return _NCOMP.get(str(cs))


def _complete_raw(data: bytes, row_bytes: int, h: int) -> tuple[bytes, int]:
    """Make `data` exactly h rows of row_bytes: pad a short buffer by replicating the last complete row (a partial
    last row is completed from the row above), truncate a long one.  Returns (bytes, rows_padded; negative = surplus)."""
    need = row_bytes * h
    if len(data) >= need:
        return data[:need], -((len(data) - need + row_bytes - 1) // row_bytes)
    full, rem = divmod(len(data), row_bytes)
    if full == 0:
        raise ValueError("image has no complete pixel row")
    last = data[(full - 1) * row_bytes: full * row_bytes]
    head = data
    if rem:
        head = data + last[rem:]                 # complete the partial row from the row above
        full += 1
    missing = h - full
    return head + last * missing, h - (len(data) // row_bytes)


def _repaired_raw_image(img, w: int, h: int, where: str):
    """None if the raw/Flate/RunLength/... image data has exactly the length its dictionary implies (normal case).
    Otherwise (owner Pdf, stream): a standalone, unfiltered copy with the data padded/truncated to w x h (and a WARNING)."""
    if any(f in _NON_RAW_FILTERS for f in _filters(img)):
        return None
    n = _ncomp(img)
    bpc = 1 if img.get("/ImageMask", False) else int(img.get("/BitsPerComponent", 0) or 0)
    if not n or bpc not in (1, 2, 4, 8, 16):
        return None
    try:
        data = bytes(img.read_bytes())
    except Exception:  # noqa: BLE001  (let the regular path report it)
        return None
    row_bytes = (w * n * bpc + 7) // 8
    if len(data) == row_bytes * h:
        return None
    fixed, rows = _complete_raw(data, row_bytes, h)
    if rows >= 0:
        log.warning("%s: image data is short (%d of %d bytes); padded %d of %d rows by replicating the last complete row",
                    where, len(data), row_bytes * h, rows, h)
    else:
        log.warning("%s: image data is too long (%d bytes, %d expected); extra data ignored", where, len(data), row_bytes * h)
    pdf = pikepdf.Pdf.new()
    st = pdf.make_stream(fixed)
    for k in ("/Width", "/Height", "/BitsPerComponent", "/ImageMask", "/Decode", "/Intent", "/Interpolate"):
        if k in img:
            st[k] = img[k]
    if "/ColorSpace" in img:
        st["/ColorSpace"] = pdf.copy_foreign(img["/ColorSpace"]) if img["/ColorSpace"].is_indirect else img["/ColorSpace"]
    st.Type, st.Subtype = Name.XObject, Name.Image
    return pdf, st                       # the caller must keep `pdf` alive while using `st`


def _decode_truncated_jpeg(raw: bytes, w: int, h: int, where: str) -> np.ndarray:
    """Decode a JPEG whose data is cut short (no EOI): Pillow's truncation tolerance, scoped to this call.
    Rows that were not decoded (Pillow/libjpeg fill them flat gray 128) are replaced by the last decoded row."""
    from PIL import Image, ImageFile
    prev = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(io.BytesIO(raw)) as im:
            arr = np.array(im.convert("RGB"))
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = prev
    if arr.shape[1] != w:
        raise ValueError(f"truncated JPEG is {arr.shape[1]} px wide, dictionary says {w}")
    flat = (arr == 128).all(axis=(1, 2))
    valid = arr.shape[0]
    while valid > 0 and flat[valid - 1]:
        valid -= 1
    if valid < arr.shape[0] or arr.shape[0] < h:
        valid = max(valid - 16, 1)                # the last MCU row may be partly decoded: drop it too (<= 16 px)
        arr = arr[:valid]
        pad = h - valid
        arr = np.concatenate([arr, np.repeat(arr[-1:], pad, axis=0)])
        log.warning("%s: JPEG data is truncated (%d bytes); padded %d of %d rows by replicating the last decoded row",
                    where, len(raw), pad, h)
    else:
        log.warning("%s: JPEG data has no end marker (%d bytes) but decoded completely", where, len(raw))
    return arr[:h]


def extract_page_input(page, index: int, name: str = "") -> PageInput:
    img = next(iter(page.resources.XObject.values()))
    w, h = int(img.Width), int(img.Height)
    box = page_media_box(page)
    dpi = detect_dpi(w, h, box)
    rot = page_rotate(page)
    where = f"{name + ': ' if name else ''}page {index + 1}"
    if _plain_dct(img):
        raw = bytes(img.read_raw_bytes())
        if raw[-64:].rstrip(b"\x00")[-2:] == b"\xff\xd9":
            return PageInput(index, "jpeg", raw, (w, h), box, rot, dpi)
        arr = _decode_truncated_jpeg(raw, w, h, where)
        return PageInput(index, "raw", arr.tobytes(), (arr.shape[1], arr.shape[0]), box, rot, dpi)
    # general path (Flate, RunLength, CMYK, 1-bit, indexed, ...): let pikepdf decode
    fixed = _repaired_raw_image(img, w, h, where)
    pil = PdfImage(img if fixed is None else fixed[1]).as_pil_image()
    arr = np.asarray(pil.convert("RGB"))
    return PageInput(index, "raw", arr.tobytes(), (arr.shape[1], arr.shape[0]), box, rot, dpi)


def iter_page_inputs(pdf: pikepdf.Pdf, indices=None, name: str = "") -> Iterator[PageInput]:
    """Lazily yield PageInputs (one at a time so memory stays bounded)."""
    for i in (range(len(pdf.pages)) if indices is None else indices):
        yield extract_page_input(pdf.pages[i], i, name)


# ----------------------------------------------------------------------------- writing
def build_metadata(src_info: dict | None, stem: str) -> dict:
    """Keep Creator/Producer of the source, append scanpp to Producer, title = file stem."""
    src_info = src_info or {}
    producer = str(src_info.get("/Producer", "") or "").strip()
    tag = f"scanpp {__version__}"
    if "scanpp" not in producer:
        producer = f"{producer}; {tag}" if producer else tag
    meta = {"title": stem, "producer": producer}
    creator = src_info.get("/Creator")
    if creator is not None and str(creator):
        meta["creator"] = str(creator)
    return meta


def _make_image(pdf: pikepdf.Pdf, spec: ImageSpec) -> pikepdf.Stream:
    st = pikepdf.Stream(pdf, spec.data)
    st.Type = Name.XObject
    st.Subtype = Name.Image
    st.Width = spec.width
    st.Height = spec.height
    st.BitsPerComponent = spec.bpc
    if spec.colorspace:
        st.ColorSpace = Name("/" + spec.colorspace)
    if spec.filter:
        st.Filter = Name("/" + spec.filter)
    for k, v in spec.extra.items():
        if isinstance(v, dict):
            v = pikepdf.Dictionary({"/" + a: b for a, b in v.items()})
        elif isinstance(v, (list, tuple)):
            v = pikepdf.Array(v)
        st["/" + k] = v
    if spec.mask is not None:
        st.Mask = _make_image(pdf, spec.mask)
    if spec.smask is not None:
        st.SMask = _make_image(pdf, spec.smask)
    return st


def write_pdf(pages: list[EncodedPage], path, metadata: dict | None = None) -> None:
    """Write EncodedPages to `path`. Each page keeps its source MediaBox and /Rotate."""
    metadata = metadata or {}
    pdf = pikepdf.Pdf.new()
    font_cache: dict = {}
    for ep in pages:
        x0, y0, x1, y1 = ep.media_box
        xo = pikepdf.Dictionary({"/" + n: _make_image(pdf, s) for n, s in ep.xobjects.items()})
        content = ep.content
        if ep.ocr is not None:
            try:
                form, ops = ocr.embed_layer(pdf, ep.ocr, ep.media_box, ep.rotate, font_cache)
                xo["/" + ocr.FORM_NAME] = form
                content = (content or b"") + b"\n" + ops
            except Exception as e:  # noqa: BLE001
                log.warning("could not embed the OCR text layer of a page (%s); page written without text", e)
        if (x0 or y0) and content:
            content = f"q 1 0 0 1 {x0:.4f} {y0:.4f} cm\n".encode() + content + b"Q\n"
        page = pikepdf.Dictionary(
            Type=Name.Page,
            MediaBox=pikepdf.Array([x0, y0, x1, y1]),
            Resources=pikepdf.Dictionary(XObject=xo),
            Contents=pikepdf.Stream(pdf, content),
        )
        page[MARK_PAGE] = __version__
        if ep.rotate:
            page.Rotate = ep.rotate
        pdf.pages.append(pikepdf.Page(page))
    info = pdf.docinfo
    info[MARK_DOC] = __version__
    if metadata.get("title"):
        info["/Title"] = metadata["title"]
    if metadata.get("creator"):
        info["/Creator"] = metadata["creator"]
    if metadata.get("producer"):
        info["/Producer"] = metadata["producer"]
    with pdf.open_metadata(set_pikepdf_as_editor=False, update_docinfo=False) as xmp:   # XMP mirror of the marker
        xmp["xmp:CreatorTool"] = f"scanpp {__version__}"
        if metadata.get("title"):
            xmp["dc:title"] = metadata["title"]
    pdf.save(str(path), min_version="1.5")


def validate_pdf(path, expected_pages: int) -> None:
    """Raise ValueError unless `path` opens, has `expected_pages` pages and every page carries an image."""
    with pikepdf.open(str(path)) as pdf:
        if len(pdf.pages) != expected_pages:
            raise ValueError(f"output has {len(pdf.pages)} pages, expected {expected_pages}")
        for i, page in enumerate(pdf.pages):
            xo = page.obj.get("/Resources", {}).get("/XObject", {})
            if len(xo) == 0 and len(page.obj.Contents.read_bytes().strip()) == 0:
                continue                                        # intentionally blank page: empty content stream
            if not any(v.get("/Subtype") == Name.Image for v in xo.values()):
                raise ValueError(f"output page {i + 1} has no image")
