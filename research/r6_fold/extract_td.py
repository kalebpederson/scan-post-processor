"""Decode the pages of a private multi-page scan with fold creases (usage: extract_td.py INPUT.pdf) to work/td/ as .npy (300 ppi RGB).
Page 8's Flate image is 3 rows short: it is decoded by hand and the last row replicated (poppler pads with black).
Private data: the outputs never go into the repo."""
import zlib, numpy as np, pikepdf, cv2
from pikepdf import PdfImage
from common import SCR, ROOT
import os
os.makedirs(SCR + '/td', exist_ok=True)
import sys
if len(sys.argv) != 2:
    sys.exit('usage: extract_td.py INPUT.pdf')
pdf = pikepdf.open(sys.argv[1])
for i, p in enumerate(pdf.pages):
    name, raw = list(p.images.items())[0]
    W, H = int(raw.Width), int(raw.Height)
    try:
        a = np.asarray(PdfImage(raw).as_pil_image().convert('RGB'))
        if a.shape[0] != H:
            raise ValueError
    except Exception:
        try:
            b = raw.read_bytes()
        except Exception:
            b = zlib.decompressobj().decompress(raw.read_raw_bytes())
        arr = np.frombuffer(b, np.uint8)
        rows = len(arr) // (W * 3)
        a = arr[:rows * W * 3].reshape(rows, W, 3)
        if rows < H:
            a = np.concatenate([a, np.repeat(a[-1:], H - rows, 0)])
    np.save(f'{SCR}/td/p{i + 1:02d}.npy', a)
