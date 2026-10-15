"""Cheap halftone detector on 0.2 in tiles.  For each tile: power spectrum (Hann window, DC removed); `frac` = power in the single
strongest 3x3 spectral neighbourhood inside the 90-220 lpi band divided by total power above 15 lpi.  Halftone dots give a
narrow spectral peak (high frac); glyph edges / noise spread power over frequency (low frac).  Tiles must also be light-to-mid
(mean darkness 8-110, <3% of pixels darker than 150).  Usage: detect.py IMG DPI -> top tiles."""
import sys
import cv2, numpy as np

def tiles(gray, dpi, thr=0.12):
    t = int(round(0.2 * dpi)); win = np.outer(np.hanning(t), np.hanning(t)).astype(np.float32)
    fy = np.fft.fftfreq(t)[:, None]; fx = np.fft.fftfreq(t)[None, :]
    f = np.hypot(fx, fy) * dpi
    band = (f >= 90) & (f <= 220); hi = f >= 15
    out = []
    H, W = gray.shape
    for y in range(0, H - t, t // 2):
        for x in range(0, W - t, t // 2):
            p = gray[y:y + t, x:x + t].astype(np.float32); dk = 255 - p; m = dk.mean()
            if m < 8 or m > 110 or (dk > 150).mean() > 0.03: continue
            F = np.abs(np.fft.fft2((p - p.mean()) * win)) ** 2
            tot = F[hi].sum()
            Fb = np.where(band, F, 0)
            S = cv2.boxFilter(Fb.astype(np.float32), -1, (3, 3), normalize=False, borderType=cv2.BORDER_WRAP)
            s = float(S.max() / (tot + 1e-9))
            if s > thr: out.append((x, y, s, float(m)))
    return out, t
if __name__ == "__main__":
    im = cv2.imread(sys.argv[1], 0); dpi = float(sys.argv[2])
    if len(sys.argv) > 3: im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    o, t = tiles(im, dpi, 0.05)
    print(len(o), t)
    for r in sorted(o, key=lambda r: -r[2])[:12]: print(r)
