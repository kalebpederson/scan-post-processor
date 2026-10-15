"""Synthetic page builders and PDF writers shared by the tests."""
from __future__ import annotations

import io
import zlib
from pathlib import Path

import cv2
import numpy as np
import pikepdf
from PIL import Image
from pikepdf import Name

W, H, DPI = 850, 1100, 100        # letter at 100 dpi (small = fast)


def text_page(paper=(255, 255, 255), ink=(0, 0, 0), lines=26, angle=0.0, w=W, h=H, seed=0, noise=0.0):
    """Paper-coloured page with justified-looking lines of text; optionally rotated by `angle` degrees (CCW)."""
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = paper
    words = ["Algebra", "triangle", "the", "measure", "of", "angle", "ABC", "equals", "30", "degrees", "because",
             "parallel", "lines", "x=4"]
    y = 90
    for _ in range(lines):
        x = 70
        while x < w - 200:
            wd = words[rng.integers(len(words))]
            cv2.putText(img, wd, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ink, 1, cv2.LINE_AA)
            x += int(cv2.getTextSize(wd, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]) + 14
        y += 34 if y < h - 100 else 0
        if y > h - 60:
            break
    if noise:
        img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    if angle:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=paper)
    return img


def blank_page(paper=(250, 250, 248), specks=12, seed=1, w=W, h=H):
    rng = np.random.default_rng(seed)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = paper
    img = np.clip(img.astype(np.float32) + rng.normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)
    for _ in range(specks):
        cv2.circle(img, (int(rng.integers(60, w - 60)), int(rng.integers(60, h - 60))), 1, (150, 150, 150), -1)
    return img


def jpeg_bytes(arr, q=90):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=q)
    return buf.getvalue()


def make_scan_pdf(path, pages, dpi=DPI, kind="jpeg", rotate=0, creator="Simple Scan 48.1"):
    """simple-scan-like PDF: one image per page, content 'q W 0 0 H 0 0 cm /Im0 Do Q'.
    kind: jpeg (DCT RGB or gray) | flate | flate_gray | cmyk | bilevel | runlength."""
    pdf = pikepdf.Pdf.new()
    for arr in pages:
        h, w = arr.shape[:2]
        if kind == "jpeg":
            g = arr.ndim == 2
            st = pikepdf.Stream(pdf, jpeg_bytes(arr))
            st.Filter, st.ColorSpace = Name.DCTDecode, (Name.DeviceGray if g else Name.DeviceRGB)
            st.BitsPerComponent = 8
        elif kind in ("flate", "flate_gray"):
            data = arr if kind == "flate" else cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            st = pikepdf.Stream(pdf, data.tobytes())
            st.write(zlib.compress(data.tobytes()), filter=Name.FlateDecode)
            st.ColorSpace = Name.DeviceRGB if kind == "flate" else Name.DeviceGray
            st.BitsPerComponent = 8
        elif kind == "cmyk":
            cmyk = np.asarray(Image.fromarray(arr).convert("CMYK"))
            st = pikepdf.Stream(pdf, b"")
            st.write(zlib.compress(cmyk.tobytes()), filter=Name.FlateDecode)
            st.ColorSpace, st.BitsPerComponent = Name.DeviceCMYK, 8
        elif kind == "bilevel":
            bw = np.asarray(Image.fromarray(arr).convert("L").convert("1"))
            st = pikepdf.Stream(pdf, b"")
            st.write(zlib.compress(np.packbits(bw, axis=1).tobytes()), filter=Name.FlateDecode)
            st.ColorSpace, st.BitsPerComponent = Name.DeviceGray, 1
        elif kind == "runlength":
            raw = arr.tobytes()
            out = bytearray()
            for i in range(0, len(raw), 128):          # literal runs only (valid RunLength)
                chunk = raw[i:i + 128]
                out.append(len(chunk) - 1)
                out += chunk
            out.append(128)
            st = pikepdf.Stream(pdf, bytes(out))
            st.Filter, st.ColorSpace, st.BitsPerComponent = Name.RunLengthDecode, Name.DeviceRGB, 8
        else:
            raise ValueError(kind)
        st.Type, st.Subtype, st.Width, st.Height = Name.XObject, Name.Image, w, h
        wp, hp = w * 72.0 / dpi, h * 72.0 / dpi
        page = pikepdf.Dictionary(
            Type=Name.Page, MediaBox=[0, 0, wp, hp],
            Resources=pikepdf.Dictionary(XObject=pikepdf.Dictionary(Im0=st)),
            Contents=pikepdf.Stream(pdf, f"q\n{wp:.4f} 0 0 {hp:.4f} 0 0 cm\n/Im0 Do\nQ".encode()))
        if rotate:
            page.Rotate = rotate
        pdf.pages.append(pikepdf.Page(page))
    pdf.docinfo["/Creator"] = creator
    pdf.docinfo["/Producer"] = "cairo 1.18"
    pdf.save(str(path))
    return Path(path)


def make_digital_pdf(path):
    pdf = pikepdf.Pdf.new()
    page = pikepdf.Dictionary(
        Type=Name.Page, MediaBox=[0, 0, 612, 792],
        Resources=pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=pikepdf.Dictionary(
            Type=Name.Font, Subtype=Name.Type1, BaseFont=Name.Helvetica))),
        Contents=pikepdf.Stream(pdf, b"BT /F1 12 Tf 72 700 Td (Hello) Tj ET"))
    pdf.pages.append(pikepdf.Page(page))
    pdf.save(str(path))
    return Path(path)
