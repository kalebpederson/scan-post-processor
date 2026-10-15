"""Round-2 metrics: colour identity per stroke component, highlighter fringe error, pencil weight."""
import json
import numpy as np, cv2
from scipy import ndimage
from skimage.color import rgb2lab, deltaE_ciede2000
from config import RESEARCH
from metrics import fit

def _lab(px): return rgb2lab(px.reshape(1, -1, 3) / 255.0)[0]
def _hue(a, b): return (np.degrees(np.arctan2(b, a)) + 360) % 360
def _hdiff(h1, h2): d = np.abs(h1 - h2) % 360; return np.minimum(d, 360 - d)

class RefInfo:
    """everything derived from the reference once per page"""
    def __init__(self, name, ref):
        self.ref = ref; mn = ref.min(-1).astype(np.int16); mx = ref.max(-1).astype(np.int16)
        colored = ((mx - mn) > 40) & (mn < 200)
        ys, xs = np.nonzero(colored)
        lab = _lab(ref[ys, xs]); C = np.hypot(lab[:, 1], lab[:, 2]); ok = C > 25
        ys, xs, lab, C = ys[ok], xs[ok], lab[ok], C[ok]
        hue = _hue(lab[:, 1], lab[:, 2]); hb = (hue // 30).astype(int)
        L = np.zeros(ref.shape[:2], np.int32); nxt = 1
        for b in range(12):
            m = np.zeros(ref.shape[:2], np.uint8); sel = hb == b; m[ys[sel], xs[sel]] = 1
            if not sel.any(): continue
            n, lb = cv2.connectedComponents(m, connectivity=8)
            L[lb > 0] = lb[lb > 0] + nxt - 1; nxt += n - 1
        lab_px = L[ys, xs]; area = np.bincount(lab_px, minlength=nxt)
        d = 255 - mn[ys, xs]
        core = d >= 100
        # per component: use core pixels if >= 10 else all
        ncore = np.bincount(lab_px[core], minlength=nxt)
        use = core | (ncore[lab_px] < 10)
        good = area >= 30
        use &= good[lab_px]
        self.ys, self.xs, self.lab_px = ys[use], xs[use], lab_px[use]
        self.a, self.b = lab[use, 1], lab[use, 2]
        self.ids = np.unique(self.lab_px); self.area = area
        if len(self.ids):
            self.ref_a = ndimage.median(self.a, self.lab_px, self.ids); self.ref_b = ndimage.median(self.b, self.lab_px, self.ids)
        else: self.ref_a = self.ref_b = np.zeros(0)
        self.ref_h = _hue(np.asarray(self.ref_a), np.asarray(self.ref_b)); self.ref_C = np.hypot(self.ref_a, self.ref_b)
        self.L = L
        # highlighter regions
        pale = ((mx - mn) >= 25) & (mn >= 100) & (mn < 250)
        reg = cv2.morphologyEx(pale.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        reg = cv2.morphologyEx(reg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0
        self.hl = None
        if reg.sum() >= 5000:
            hlmin = np.median(mn[reg & pale]); ink = reg & (mn < hlmin - 60)
            near = cv2.dilate(ink.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            self.hl = dict(ink=ink, near=reg & near & ~ink & pale, hl=reg & pale & ~near)
        # pencil boxes (r2 labels, pencil class), clean-page coords
        self.pencil = []
        for l in json.load(open(RESEARCH / "r2_cleaning" / "labels.json"))["pos"]:
            if l["cls"] != "pencil": continue
            f = l["file"].split("/")[-1]; nm = "epson300" if f.startswith("epson") else f.rsplit(".", 1)[0]
            if nm != name: continue
            x, y, w, h = l["box"]; s = 300 / l["dpi"]
            if nm == "epson300": x, y, w, h = 3508 - (y + h), x, h, w; s = 1
            self.pencil.append(tuple(int(round(v * s)) for v in (x, y, w, h)))

def evaluate(info, ren):
    ren = fit(ren, info.ref.shape); out = {}
    # colour identity
    if len(info.ids):
        rl = _lab(ren[info.ys, info.xs])
        ra = np.asarray(ndimage.median(rl[:, 1], info.lab_px, info.ids)); rb = np.asarray(ndimage.median(rl[:, 2], info.lab_px, info.ids))
        rh = _hue(ra, rb); rC = np.hypot(ra, rb)
        dh = _hdiff(rh, info.ref_h); chg = (dh > 20) | (rC < 0.6 * info.ref_C)
        ar = info.area[info.ids].astype(float)
        out.update(cid_n=int(len(info.ids)), cid_changed=float(chg.mean()), cid_area_changed=float((ar * chg).sum() / ar.sum()),
                   cid_hue=float((dh > 20).mean()), cid_chroma=float((rC < 0.6 * info.ref_C).mean()))
        big = ar >= 100
        out.update(cid100_n=int(big.sum()), cid100_changed=float(chg[big].mean()) if big.any() else None, cid100_hue=float((dh > 20)[big].mean()) if big.any() else None)
        # adjacent-swap: changed component whose rendered hue matches a different-hue neighbour's reference hue
        idx = {int(i): k for k, i in enumerate(info.ids)}; swaps = 0
        objs = ndimage.find_objects(info.L)
        for k in np.nonzero(dh > 20)[0]:
            cid = int(info.ids[k]); sl = objs[cid - 1] if cid - 1 < len(objs) else None
            if sl is None: continue
            y0, y1 = max(0, sl[0].start - 6), sl[0].stop + 6; x0, x1 = max(0, sl[1].start - 6), sl[1].stop + 6
            crop = info.L[y0:y1, x0:x1]; mine = (crop == cid).astype(np.uint8)
            nb = np.unique(crop[(cv2.dilate(mine, np.ones((9, 9), np.uint8)) > 0) & (crop != cid) & (crop > 0)])
            for n in nb:
                j = idx.get(int(n))
                if j is not None and _hdiff(info.ref_h[j], info.ref_h[k]) > 20 and _hdiff(rh[k], info.ref_h[j]) <= 20: swaps += 1; break
        out["cid_swap"] = swaps / len(info.ids)
    # highlighter fringe
    if info.hl is not None:
        for key in ("ink", "near", "hl"):
            m = info.hl[key]; ys, xs = np.nonzero(m)
            if len(ys) > 60000: s = np.random.default_rng(0).choice(len(ys), 60000, replace=False); ys, xs = ys[s], xs[s]
            out["hl_" + key] = float(deltaE_ciede2000(_lab(info.ref[ys, xs]), _lab(ren[ys, xs])).mean()) if len(ys) else None
    # pencil weight
    if info.pencil:
        gr = cv2.cvtColor(info.ref, cv2.COLOR_RGB2GRAY).astype(float); gn = cv2.cvtColor(ren, cv2.COLOR_RGB2GRAY).astype(float)
        num = den = 0; rs = []
        for x, y, w, h in info.pencil:
            a = (255 - gr[y:y + h, x:x + w]).sum(); b = (255 - gn[y:y + h, x:x + w]).sum(); num += b; den += a; rs.append(b / max(a, 1))
        out["pencil_ratio"] = num / max(den, 1); out["pencil_min"] = min(rs)
    return out
