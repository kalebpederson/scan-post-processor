"""Encoders: encode(clean_rgb_300dpi, **params) -> Page (image XObjects + content stream). build_pdf() writes pages with pikepdf.
All page geometry: MediaBox = pixels * 72/300 pt, so a 300 dpi render has exactly the reference pixel grid.

Filters used: DCTDecode (baseline, sequential, Huffman-optimised), FlateDecode (+PNG predictor 15), CCITTFaxDecode (K=-1),
JBIG2Decode (generic region, lossless, NO symbol mode). """
import io, os, subprocess, tempfile, zlib, time, struct
import numpy as np, cv2
from PIL import Image
import pikepdf
from pikepdf import Name, Dictionary, Array
from scipy.spatial import cKDTree
from config import TMP
JBIG2 = os.environ.get("JBIG2", "jbig2")

class Page:
    def __init__(self, w_px, h_px, ppi=300):
        self.w, self.h = w_px, h_px; self.ppi = ppi
        self.images = []      # list of dict(name, data, dict(PDF keys), mask_of=None)
        self.content = []
        self.meta = {}
    @property
    def wpt(self): return self.w * 72.0 / self.ppi
    @property
    def hpt(self): return self.h * 72.0 / self.ppi
    def add(self, name, data, **keys):
        self.images.append(dict(name=name, data=data, keys=keys)); return name
    def draw(self, name, fill=None):
        s = ""
        if fill is not None: s += "%.4f %.4f %.4f rg " % tuple(c / 255.0 for c in fill)
        s += "q %.4f 0 0 %.4f 0 0 cm /%s Do Q\n" % (self.wpt, self.hpt, name)
        self.content.append(s)
    def payload(self): return sum(len(i["data"]) for i in self.images)

def _stream(pdf, data, keys):
    s = pdf.make_stream(data)
    s.Type = Name.XObject; s.Subtype = Name.Image
    for k, v in keys.items():
        if k == "_mask": s.Mask = v
        elif k == "_filter": s.Filter = v
        elif k == "_parms": s.DecodeParms = v
        elif k == "Interpolate": s.Interpolate = v
        else: s[Name("/" + k)] = v
    return s

def add_page(pdf, page):
    xo = Dictionary(); made = {}
    for im in page.images:
        keys = dict(im["data_keys"]) if "data_keys" in im else dict(im["keys"])
        if "_maskname" in keys:
            keys["_mask"] = made[keys.pop("_maskname")]
        s = _stream(pdf, im["data"], keys)
        made[im["name"]] = s; xo[Name("/" + im["name"])] = s
    cs = pdf.make_stream("".join(page.content).encode())
    pg = pikepdf.Page(pdf.make_indirect(Dictionary(Type=Name.Page, MediaBox=Array([0, 0, round(page.wpt, 3), round(page.hpt, 3)]),
                                                   Resources=Dictionary(XObject=xo), Contents=cs)))
    pdf.pages.append(pg)

def build_pdf(pages, path):
    pdf = pikepdf.new()
    for p in pages: add_page(pdf, p)
    pdf.save(path, object_stream_mode=pikepdf.ObjectStreamMode.generate, compress_streams=True, deterministic_id=True)

def overhead_bytes(tmp=None):
    """size of a 1-page PDF with an empty content stream (what the container itself costs)"""
    p = Page(2544, 3295); p.content.append("")
    pth = os.path.join(TMP, "_empty.pdf"); build_pdf([p], pth); return os.path.getsize(pth)

# ---------------- raw encoders ----------------
def jpeg_bytes(arr, q, sub):
    """arr uint8 HxW or HxWx3. sub: 0=4:4:4, 2=4:2:0. Baseline sequential, optimised Huffman, no progressive."""
    im = Image.fromarray(arr)
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q, subsampling=sub if arr.ndim == 3 else 0, optimize=True, progressive=False)
    return b.getvalue()

def jpeg_image(page, name, arr, q, sub, **extra):
    gray = arr.ndim == 2
    h, w = arr.shape[:2]
    data = jpeg_bytes(arr, q, sub)
    page.add(name, data, Width=w, Height=h, ColorSpace=Name.DeviceGray if gray else Name.DeviceRGB, BitsPerComponent=8,
             _filter=Name.DCTDecode, **extra)

def _paeth(a, b, c):
    p = a + b - c; pa = np.abs(p - a); pb = np.abs(p - b); pc = np.abs(p - c)
    return np.where((pa <= pb) & (pa <= pc), a, np.where(pb <= pc, b, c))

def png_predict(idx, bpc):
    """idx: HxW uint8 palette indices. Returns zlib stream with PNG per-row filters (Predictor 15) for given bit depth."""
    h, w = idx.shape
    if bpc < 8:
        per = 8 // bpc; wp = -(-w // per) * per
        a = np.zeros((h, wp), np.uint8); a[:, :w] = idx
        a = a.reshape(h, -1, per); sh = (np.arange(per)[::-1] * bpc).astype(np.uint8)
        rows = np.bitwise_or.reduce(a << sh, axis=2).astype(np.uint8)
    else: rows = idx
    x = rows.astype(np.int16)
    left = np.zeros_like(x); left[:, 1:] = x[:, :-1]
    up = np.zeros_like(x); up[1:] = x[:-1]
    ul = np.zeros_like(x); ul[1:, 1:] = x[:-1, :-1]
    cands = [x, x - left, x - up, x - ((left + up) >> 1), x - _paeth(left, up, ul)]
    cands = [(c & 255).astype(np.uint8) for c in cands]
    cost = np.stack([np.abs(np.where(c > 127, c.astype(np.int16) - 256, c.astype(np.int16))).sum(1) for c in cands])
    best = cost.argmin(0)
    out = np.empty((h, rows.shape[1] + 1), np.uint8)
    out[:, 0] = best
    stack = np.stack(cands)           # 5,h,w
    out[:, 1:] = stack[best, np.arange(h)]
    raw = out.tobytes()
    bestz = None
    for strat in (zlib.Z_DEFAULT_STRATEGY, zlib.Z_FILTERED):
        c = zlib.compressobj(9, zlib.DEFLATED, 15, 9, strat); z = c.compress(raw) + c.flush()
        if bestz is None or len(z) < len(bestz): bestz = z
    return bestz

def indexed_image(page, name, idx, palette_rgb, gray=False):
    h, w = idx.shape; k = len(palette_rgb)
    bpc = 1 if k <= 2 else 2 if k <= 4 else 4 if k <= 16 else 8
    data = png_predict(idx, bpc)
    pal = np.asarray(palette_rgb, np.uint8)
    if gray: base = Name.DeviceGray; palb = pal[:, 0].tobytes()
    else: base = Name.DeviceRGB; palb = pal.reshape(-1).tobytes()
    cs = Array([Name.Indexed, base, k - 1, pikepdf.String(palb)])
    page.add(name, data, Width=w, Height=h, ColorSpace=cs, BitsPerComponent=bpc, _filter=Name.FlateDecode,
             _parms=Dictionary(Predictor=15, Colors=1, BitsPerComponent=bpc, Columns=w))
    return bpc

def mask_bytes(ink, codec):
    """ink: HxW bool (True = ink). Returns (data, keys) for a stencil. Sample 0 = painted (ink), default Decode."""
    h, w = ink.shape
    if codec == "jbig2":
        fd, p = tempfile.mkstemp(suffix=".png", dir=TMP); os.close(fd)
        cv2.imwrite(p, np.where(ink, 0, 255).astype(np.uint8))
        out = subprocess.run([JBIG2, "-p", "-d", "-G", p], capture_output=True, check=True).stdout   # generic region only (no -s!)
        os.remove(p)
        return out, dict(_filter=Name.JBIG2Decode)
    if codec == "g4":
        im = Image.fromarray(np.where(ink, 0, 255).astype(np.uint8)).convert("1")
        b = io.BytesIO(); im.save(b, "TIFF", compression="group4", tiffinfo={278: h}); import tifffile
        with tifffile.TiffFile(io.BytesIO(b.getvalue())) as tf:
            pg = tf.pages[0]; assert len(pg.dataoffsets) == 1
            raw = b.getvalue()[pg.dataoffsets[0]: pg.dataoffsets[0] + pg.databytecounts[0]]
            photo = pg.photometric
        # PIL writes WhiteIsZero? verify by decode-on-render in tests; BlackIs1 follows photometric
        bi1 = True
        return raw, dict(_filter=Name.CCITTFaxDecode, _parms=Dictionary(K=-1, Columns=w, Rows=h, BlackIs1=bi1))
    if codec == "flate":
        pk = np.packbits(~ink, axis=1)
        return zlib.compress(pk.tobytes(), 9), dict(_filter=Name.FlateDecode)
    raise ValueError(codec)

def stencil(page, name, ink, codec, **extra):
    data, keys = mask_bytes(ink, codec)
    h, w = ink.shape
    page.add(name, data, Width=w, Height=h, ImageMask=True, BitsPerComponent=1, **keys, **extra)

# ---------------- E1: JPEG ----------------
def resize_to(rgb, ppi):
    h, w = rgb.shape[:2]
    if ppi == 300: return rgb
    return cv2.resize(rgb, (round(w * ppi / 300), round(h * ppi / 300)), interpolation=cv2.INTER_AREA)

def to_gray(rgb): return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

def enc_jpeg(rgb, r=200, q=70, sub=2, gray=False):
    h, w = rgb.shape[:2]; pg = Page(w, h)
    img = resize_to(rgb, r)
    if gray: img = to_gray(img)
    jpeg_image(pg, "Im0", img, q, sub); pg.draw("Im0"); return pg

# ---------------- E2: palette + Flate ----------------
def _lab(rgb_u8):
    return cv2.cvtColor(rgb_u8.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

def make_palette(pix, k, method):
    """pix: Nx3 uint8 (non-white pixels). returns (k-1)x3 uint8 colours."""
    if len(pix) == 0: return np.zeros((0, 3), np.uint8)
    k = min(k, len(np.unique(pix, axis=0)))
    if method == "kmeans":
        lab = _lab(pix)
        rng = np.random.default_rng(0)
        if len(lab) > 150000: sel = rng.choice(len(lab), 150000, replace=False); lab_s = lab[sel]; pix_s = pix[sel]
        else: lab_s = lab; pix_s = pix
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, lbl, cen = cv2.kmeans(lab_s, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        lbl = lbl.ravel(); out = []
        for i in range(k):
            m = lbl == i
            if m.any(): out.append(pix_s[m].astype(np.float32).mean(0))
        return np.clip(np.array(out) + 0.5, 0, 255).astype(np.uint8)
    if method == "mediancut":
        n = len(pix); side = int(np.ceil(np.sqrt(n)))
        a = np.zeros((side * side, 3), np.uint8); a[:n] = pix; a[n:] = pix[0]
        im = Image.fromarray(a.reshape(side, side, 3)).quantize(colors=k, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        pal = np.array(im.getpalette()[:3 * k], np.uint8).reshape(-1, 3)
        used = np.unique(np.array(im).ravel()); return pal[used]
    raise ValueError(method)

def assign(rgb, pal_rgb, white_idx=0):
    """map every pixel to nearest palette entry (Lab); exact-white pixels -> white_idx. pal includes white at white_idx."""
    h, w = rgb.shape[:2]
    flat = rgb.reshape(-1, 3)
    key = (flat[:, 0].astype(np.uint32) << 16) | (flat[:, 1].astype(np.uint32) << 8) | flat[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    ucol = np.stack([(uk >> 16) & 255, (uk >> 8) & 255, uk & 255], 1).astype(np.uint8)
    tree = cKDTree(_lab(pal_rgb))
    _, ui = tree.query(_lab(ucol))
    ui = ui.astype(np.uint8)
    ui[uk == 0xFFFFFF] = white_idx
    return ui[inv.reshape(-1)].reshape(h, w)

def enc_palette(rgb, r=200, k=16, method="kmeans", gray=False):
    h, w = rgb.shape[:2]; pg = Page(w, h)
    img = resize_to(rgb, r)
    if gray:
        g = to_gray(img)
        nz = g[g < 255].reshape(-1, 1)
        if len(nz) == 0: pal = np.zeros((0, 1))
        elif method == "kmeans":
            kk = min(k - 1, len(np.unique(nz)))
            _, lbl, cen = cv2.kmeans(nz[::max(1, len(nz) // 150000)].astype(np.float32), kk, None,
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2), 3, cv2.KMEANS_PP_CENTERS)
            pal = np.sort(np.clip(cen.ravel() + 0.5, 0, 254).astype(np.uint8))
        else:   # uniform levels
            pal = np.linspace(0, 255, k, endpoint=False).astype(np.uint8)[: k - 1]
        pal = np.concatenate([[255], pal]).astype(np.uint8)
        d = np.abs(g[..., None].astype(np.int16) - pal[None, None, :].astype(np.int16))
        idx = d.argmin(-1).astype(np.uint8); idx[g == 255] = 0
        indexed_image(pg, "Im0", idx, np.stack([pal] * 3, 1), gray=True)
    else:
        nz = img.reshape(-1, 3)[(img != 255).any(-1).reshape(-1)]
        pal = make_palette(nz, k - 1, method)
        pal = np.concatenate([[[255, 255, 255]], pal]).astype(np.uint8)
        idx = assign(img, pal)
        indexed_image(pg, "Im0", idx, pal)
    pg.draw("Im0"); return pg

# ---------------- ink masks / colour layers ----------------
def darkness(rgb): return 255 - rgb.min(-1).astype(np.int16)

ADAPT = (0.35, 20, 88)     # ink threshold on darkness d=255-min(rgb): clip(frac*local_max_7x7(d), lo, hi)

def ink_mask(rgb, thr, mask_ppi, bg=None):
    """binary ink at mask_ppi from the 300 ppi clean image. darkness = 255-min(rgb) (relative to bg layer if given).
    thr: number (global) or (frac, lo, hi) adaptive: the pixel is ink if d >= clip(frac*local max of d (7x7 @300), lo, hi);
    adaptive keeps very light halftone/pencil strokes (low lo) without fattening black text (frac*max) and keeps highlighter (hi<~90)."""
    mn = rgb.min(-1).astype(np.float32)
    d = (255.0 - mn) if bg is None else np.maximum(0, bg.min(-1).astype(np.float32) - mn)
    h, w = d.shape
    if mask_ppi != 300:
        sz = (round(w * mask_ppi / 300), round(h * mask_ppi / 300))
        d = cv2.resize(d, sz, interpolation=cv2.INTER_AREA if mask_ppi < 300 else cv2.INTER_CUBIC)
    if isinstance(thr, (tuple, list)):
        f, lo, hi = thr; k = max(3, int(round(7 * mask_ppi / 300)) | 1)
        t = np.clip(f * cv2.dilate(d, np.ones((k, k), np.uint8)), lo, hi)
        return d >= t
    return d >= thr

def pushpull(col, valid):
    """smooth push-pull inpainting: fill ~valid pixels of col (HxWx3 float32) from a coarse-to-fine weighted pyramid.
    Result is smooth off-ink (cheap for JPEG) and exact on valid pixels."""
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
            a = np.clip(w * 4, 0, 1)[..., None]       # confidence of this level's own estimate
            est = est * a + up * (1 - a)
        fill = est
    return np.where(valid[..., None], col, fill)

def _cell_index(h, w, ch, cw):
    ys = np.minimum((np.arange(h) * ch) // h, ch - 1); xs = np.minimum((np.arange(w) * cw) // w, cw - 1)
    return ys[:, None] * cw + xs[None, :]

def color_layer(rgb, ink300, col_ppi, gamma=3.0, mode="mean", K=12, bg=None):
    """low-res colour under the ink. Cells without ink are filled by smooth push-pull inpainting.
    mode 'mean': darkness^gamma-weighted mean of the ink pixels (normalized convolution over ink only).
    mode 'dom' : per cell, the colour of the DOMINANT ink colour cluster only (pixel-count vote over a per-page Lab k-means palette of K
                 clusters), averaged over that cluster's pixels in the cell -> adjacent strokes of different colours do not blend."""
    h, w = rgb.shape[:2]
    cw, ch = round(w * col_ppi / 300), round(h * col_ppi / 300)
    d = darkness(rgb).astype(np.float32) / 255.0
    ink = ink300 > 0
    if mode == "mean":
        wgt = (d ** gamma) * ink300
        num = cv2.resize((rgb.astype(np.float32) * wgt[..., None]), (cw, ch), interpolation=cv2.INTER_AREA)
        den = cv2.resize(wgt, (cw, ch), interpolation=cv2.INTER_AREA)
        valid = den > 1e-6
        if not valid.any(): return np.zeros((ch, cw, 3), np.uint8)
        col = num / np.maximum(den, 1e-9)[..., None]
    else:
        ys, xs = np.nonzero(ink)
        if len(ys) == 0: return np.zeros((ch, cw, 3), np.uint8)
        pix = rgb[ys, xs]; lab = _lab(pix)
        rng = np.random.default_rng(0); sel = rng.choice(len(lab), min(len(lab), 80000), replace=False)
        kk = min(K, len(sel))
        _, _, cen = cv2.kmeans(lab[sel], kk, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5), 2, cv2.KMEANS_PP_CENTERS)
        lbl = cKDTree(cen).query(lab)[1]
        cell = (np.minimum((ys * ch) // h, ch - 1) * cw + np.minimum((xs * cw) // w, cw - 1))
        votes = np.bincount(cell * kk + lbl, weights=d[ys, xs], minlength=ch * cw * kk).reshape(ch * cw, kk)
        win = votes.argmax(1)
        m = lbl == win[cell]
        wg = d[ys, xs] ** gamma * m
        den = np.bincount(cell, weights=wg, minlength=ch * cw)
        num = np.stack([np.bincount(cell, weights=wg * pix[:, c], minlength=ch * cw) for c in range(3)], 1)
        valid = (den > 1e-9).reshape(ch, cw)
        col = (num / np.maximum(den, 1e-9)[:, None]).reshape(ch, cw, 3).astype(np.float32)
    col = pushpull(col, valid) if not valid.all() else col
    return np.clip(col + 0.5, 0, 255).astype(np.uint8)

def upmask(ink_lo, size):  # ink at mask ppi -> 300 grid (nearest) for colour weighting
    return cv2.resize(ink_lo.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(np.float32)

def highlighter_layer(rgb, bg_ppi=75, min_w_px=9):
    """pale saturated wide regions (highlighter). returns (bg300 uint8 HxWx3 reconstructed highlighter-only layer, lowres bg image or None)"""
    mn = rgb.min(-1).astype(np.int16); mx = rgb.max(-1).astype(np.int16)
    pale = ((mx - mn) >= 25) & (mn >= 100) & (mn < 250)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (min_w_px, min_w_px))
    reg = cv2.morphologyEx(pale.astype(np.uint8), cv2.MORPH_OPEN, k)
    if reg.sum() < 3000 * (min_w_px / 9) ** 2: return None
    reg = cv2.morphologyEx(reg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    h, w = mn.shape; cw, ch = round(w * bg_ppi / 300), round(h * bg_ppi / 300)
    wgt = (reg.astype(np.float32)) * pale
    num = cv2.resize(rgb.astype(np.float32) * wgt[..., None], (cw, ch), interpolation=cv2.INTER_AREA)
    den = cv2.resize(wgt, (cw, ch), interpolation=cv2.INTER_AREA)
    regs = cv2.resize(reg.astype(np.float32), (cw, ch), interpolation=cv2.INTER_AREA)
    col = np.where((den > 1e-4)[..., None], num / np.maximum(den, 1e-6)[..., None], 255.0)
    # blend to white by region coverage so edges stay soft and sized right
    valid = den > 1e-4
    if valid.any() and not valid.all():
        _, lab = cv2.distanceTransformWithLabels((~valid).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(valid); lut = np.zeros(lab.max() + 1, np.int64); lut[lab[ys, xs]] = ys * cw + xs
        col = col.reshape(-1, 3)[lut[lab.reshape(-1)]].reshape(ch, cw, 3)
    a = np.clip(regs, 0, 1)[..., None]
    low = np.clip(255 * (1 - a) + col * a + 0.5, 0, 255).astype(np.uint8)
    return low

# ---------------- E3: mask + colour layer ----------------
def enc_mrc(rgb, mask_ppi=300, codec="jbig2", col_ppi=100, q=55, sub=2, thr=ADAPT, hl=False, gray=False, hl_q=60, hl_ppi=75, hl_w=9, colmode="mean", dil=0):
    h, w = rgb.shape[:2]; pg = Page(w, h)
    bg300 = None
    if hl:
        low = highlighter_layer(rgb, hl_ppi, hl_w)
        if low is not None:
            jpeg_image(pg, "Bg", low, hl_q, 2); pg.draw("Bg")
            bg300 = cv2.resize(low, (w, h), interpolation=cv2.INTER_LINEAR)
    ink = ink_mask(rgb, thr, mask_ppi, bg300)
    if not ink.any(): return pg
    if dil: ink = cv2.dilate(ink.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=dil).astype(bool)
    ink300 = ink if mask_ppi == 300 else upmask(ink, (w, h))
    col = color_layer(rgb, ink300, col_ppi, mode=colmode)
    if gray: col = to_gray(col)
    stencil(pg, "Mk", ink, codec)
    jpeg_image(pg, "Fg", col, q, sub, _maskname="Mk")
    pg.draw("Fg"); return pg

# ---------------- E4: colour-separated stencils ----------------
def enc_stencils(rgb, K=7, codec="jbig2", thr=ADAPT, mask_ppi=300, merge_de=18.0, under_dilate=1):
    h, w = rgb.shape[:2]; pg = Page(w, h)
    ink = ink_mask(rgb, thr, 300)
    if not ink.any(): return pg
    pix = rgb[ink]
    lab = _lab(pix)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(lab), min(len(lab), 120000), replace=False)
    kk = min(K, len(sel))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _, lbl, cen = cv2.kmeans(lab[sel], kk, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    # merge clusters closer than merge_de; palette colour = darkness^3-weighted mean RGB of members
    lbl_all = cKDTree(cen).query(lab)[1]
    order = np.argsort(-np.bincount(lbl_all, minlength=kk))
    keep = []
    for c in order:
        if all(np.linalg.norm(cen[c] - cen[k2]) > merge_de for k2 in keep): keep.append(c)
    cen = cen[keep]
    lbl_all = cKDTree(cen).query(lab)[1]
    pal = []
    d = (darkness(rgb)[ink].astype(np.float32) / 255.0) ** 3
    for i in range(len(cen)):
        m = lbl_all == i
        pal.append(np.clip((pix[m] * d[m, None]).sum(0) / max(d[m].sum(), 1e-9) + 0.5, 0, 255).astype(np.uint8))
    pal = np.array(pal)
    lum = pal.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    idxmap = np.full((h, w), -1, np.int16); idxmap[ink] = lbl_all
    drawn = 0
    for i in np.argsort(-lum):                      # lightest first (highlighter underneath text)
        m = idxmap == i
        if not m.any(): continue
        if under_dilate and lum[i] > lum.min() + 1e-3 and i != np.argmin(lum):
            m = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=under_dilate).astype(bool)
        nm = "M%d" % i
        if mask_ppi != 300:
            m = cv2.resize(m.astype(np.uint8), (round(w * mask_ppi / 300), round(h * mask_ppi / 300)), interpolation=cv2.INTER_AREA) > 0
        stencil(pg, nm, m, codec); pg.draw(nm, fill=tuple(int(c) for c in pal[i])); drawn += 1
    pg.meta["palette"] = pal.tolist()
    return pg


# ---------------- E5: JPEG base + sharp black text stencil ----------------
def neutral_text_mask(rgb, P=170, Cn=40, dmin=60, thr=(0.4, 40, 100)):
    """300 ppi mask of dark NEUTRAL ink (printed text, dark pen): low chroma, and the stroke component's core (d>=100 pixels) has mean
    darkness >= P (pencil strokes are lighter and stay in the JPEG). Edge threshold = clip(0.4*local max, 40, 100)."""
    from scipy import ndimage
    mn = rgb.min(-1).astype(np.int16); d = 255 - mn; chroma = rgb.max(-1).astype(np.int16) - mn
    cand = (chroma <= Cn) & (d >= dmin)
    n, lab = cv2.connectedComponents(cand.astype(np.uint8), connectivity=8)
    core = cand & (d >= 100)
    cnt = np.bincount(lab[core], minlength=n); sm = np.bincount(lab[core], weights=d[core], minlength=n)
    keep = (cnt >= 4) & (sm / np.maximum(cnt, 1) >= P); keep[0] = False
    # reject dark cores of saturated pen strokes: components whose pixels mostly sit next to coloured ink
    colnb = cv2.dilate(((chroma > 45) & (d >= 40)).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    allc = np.bincount(lab[cand], minlength=n); nbc = np.bincount(lab[cand & colnb], minlength=n)
    keep &= (nbc / np.maximum(allc, 1)) < 0.4
    dk = np.where(keep[lab], d, 0).astype(np.float32)
    t = np.clip(thr[0] * cv2.dilate(dk, np.ones((7, 7), np.uint8)), thr[1], thr[2])
    return (dk >= t) & keep[lab]

def enc_hybrid(rgb, r=150, q=40, sub=2, variant="a", P=170, Cn=40, rem_dil=1, thr=(0.4, 40, 100)):
    """E5a: JPEG(clean page) + black stencil on top.  E5b: text pixels are first replaced by local background (push-pull) in the JPEG."""
    h, w = rgb.shape[:2]; pg = Page(w, h)
    m = neutral_text_mask(rgb, P, Cn, thr=thr)
    base = rgb
    if variant == "b" and m.any():
        rem = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=rem_dil) if rem_dil else m.astype(np.uint8)
        lo = resize_to(rgb, r).astype(np.float32); lh, lw = lo.shape[:2]
        rl = cv2.resize(rem.astype(np.float32), (lw, lh), interpolation=cv2.INTER_AREA) > 0.34
        base = np.clip(pushpull(lo, ~rl) + 0.5, 0, 255).astype(np.uint8); img = base
    else: img = resize_to(rgb, r)
    jpeg_image(pg, "Im0", img, q, sub); pg.draw("Im0")
    if m.any():
        fill = tuple(int(v) for v in np.median(rgb[m & ((255 - rgb.min(-1).astype(np.int16)) >= P)], axis=0)) if (m & ((255 - rgb.min(-1).astype(np.int16)) >= P)).any() else (0, 0, 0)
        stencil(pg, "Mt", m, "jbig2"); pg.draw("Mt", fill=fill)
    pg.meta["fill"] = None
    return pg
