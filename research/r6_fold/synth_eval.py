"""Synthetic evaluation: real crease strips (transmittance) pasted over corpus pages; current clean_page vs. clean_page + fold removal.

For every instance: corpus page (native resolution) x crease strip (p4 / p6 / p14 / p13 upper fold of the test document, scaled to the page dpi)
placed at a row chosen to cross ink (probability proportional to the row's ink) or at a customary fold position, tilt in +-0.3 deg.
  raw_paste = page * T                                   (the crease darkens multiplicatively, per channel)
  V0 current   : clean(raw_paste)
  VA pre-clean : clean(remove_folds(raw_paste))          (removal on the raw image, then the unchanged cleaner)
  VC post-clean: whiten the fold mask in clean(raw_paste)
  REF          : clean(page)                              (what the page looks like without any crease)
Metrics (band = +-0.06 in around the crease; "core" = R2 stroke core of the ORIGINAL page in the band: blurred density >= max(0.04, 0.5 * p95 of the band)):
  ink retention  = fraction of core pixels with Delta-E(Lab76) to white >= 6 in the output  (REF gives the ceiling)
  ink loss       = REF retention - variant retention (same pixels)
  survival10/40  = pixels of the crease footprint (T_mean < 1 - 10/255 resp. 1 - 40/255, > 2 px from core) whose output luminance is darker than 10 / 40 gl
  extra ink px   = output pixels darker than 40 gl in the band that are paper (white) in REF
Modes: "oracle" (fold given: removal alone) and "e2e" (detect_folds on the pasted page, customary positions only).
Usage: python synth_eval.py [n_per_class] ; writes scratch/synth_results.json and research/r6_fold/synth_results.json (aggregates + per-instance numbers, no images).
"""
import json, os, sys, time
from multiprocessing import Pool
from common import *
import fold as F
from scanpp.clean import clean_page
from scanpp import geometry

P = F.FoldParams()
VARIANTS = {'stroke': F.FoldParams(orient_h_min=0.0),                    # vertical-run stroke test only (first version)
            'default': F.FoldParams(),                                    # + structure-tensor orientation test, tone_dark 0.40
            'td55': F.FoldParams(tone_dark=0.55),
            'noprotchroma': F.FoldParams(protect_chroma=False),
            'bw12': F.FoldParams(band_in=0.012)}
MAIN = 'default'
CL = json.load(open(SCR + '/classify.json')) if os.path.exists(SCR + '/classify.json') else None
STRIPS = ['p4', 'p6', 'p14', 'p13']


def load_wander(name):
    return np.load(f'{SCR}/wander_{name}.npy')


def load_strip(name, dpi):
    s = np.load(f'{SCR}/strip_{name}.npy')
    s[:, :25] = 1.0
    s[:, -25:] = 1.0
    if abs(dpi - 300) > 1:
        k = dpi / 300.0
        s = cv2.resize(s, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return s


def paste(page, strip, y0, tilt_deg, dpi, wander=None):
    """Multiply `page` by the crease transmittance whose centre line is y0 + tan(tilt) (x - W/2)."""
    h, w = page.shape[:2]
    J = (strip.shape[0] - 1) // 2
    t = np.tan(np.radians(tilt_deg))
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    wd = np.zeros(w, np.float32) if wander is None else np.interp(xs * (300.0 / dpi), np.arange(len(wander)), wander).astype(np.float32) * (dpi / 300.0)
    row = ys[:, None] - (y0 + t * (xs[None, :] - w / 2.0) + wd[None, :]) + J
    col = np.broadcast_to(np.minimum(xs, strip.shape[1] - 1)[None, :], (h, w)).astype(np.float32)
    lo, hi = max(0, int(y0 - abs(t) * w / 2 - np.abs(wd).max() - J - 2)), min(h, int(y0 + abs(t) * w / 2 + np.abs(wd).max() + J + 3))
    T = cv2.remap(strip, col[lo:hi], row[lo:hi].astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(1, 1, 1))
    out = page.copy()
    out[lo:hi] = np.clip(page[lo:hi].astype(np.float32) * T, 0, 255).astype(np.uint8)
    Tfull = np.ones((h, w), np.float32)
    Tfull[lo:hi] = T.mean(2)
    return out, Tfull


def lab_de_white(rgb):
    lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    return np.sqrt((lab[..., 0] - 100.0) ** 2 + lab[..., 1] ** 2 + lab[..., 2] ** 2)


def lum(rgb):
    return rgb.astype(np.float32).mean(2)


def run_instance(args):
    try:
        return _run_instance(args)
    except Exception as e:      # keep the sweep going; the error is reported in the results
        import traceback
        return dict(idx=args[0], strip=args[1], mode=args[2], err=traceback.format_exc()[-600:])


def _run_instance(args):
    idx, strip_name, mode, seed, cls = args
    rng = np.random.default_rng(seed)
    page, rec = native_page(idx)
    dpi = float(rec['dpi'])
    page = geometry._clean_edges_g(np.array(page), dpi, inplace=True)[0]      # pipeline order: edge clean first
    h, w = page.shape[:2]
    g = lum(page)
    dens = F.ink_density(g, dpi)
    if mode in ('prior', 'e2e'):  # a customary fold position (+- 0.01)
        yf = rng.choice(P.prior_pos) + rng.uniform(-0.01, 0.01)
        y0 = yf * h
    else:                        # a row that crosses ink
        rs = (cv2.dilate((dens > 0.15).astype(np.uint8), np.ones((int(0.12 * dpi) | 1, 1), np.uint8)).sum(1) > 0).astype(np.float64)   # rows with ink within +-0.06 in
        rs[:int(0.08 * h)] = 0
        rs[int(0.92 * h):] = 0
        y0 = float(rng.choice(h, p=rs / rs.sum())) if rs.sum() > 0 else 0.5 * h
    tilt = float(rng.uniform(-0.3, 0.3))
    strip = load_strip(strip_name, dpi)
    pasted, T = paste(page, strip, y0, tilt, dpi, load_wander(strip_name))
    fold = F.Fold(y0 / h, tilt, 0.0, 1.0, (1, 1), 99)
    # --- variants
    ref = clean_page(page, dpi)
    v0 = clean_page(pasted, dpi)
    t = time.process_time()
    if mode == 'e2e':
        folds = F.detect_folds(pasted, dpi, P)
        det = [(round(f.y_frac, 3), f.info['tier']) for f in folds]
    else:
        folds = [fold]
        det = None
    t_det = time.process_time() - t
    outs = {'v0': v0, 'ref': ref}
    boxes = {}
    if os.environ.get('R6_OCR'):
        from ocrlib import tess_words
        words, cpu_ocr = tess_words(pasted, dpi, ocr_dpi=150.0)
        for thr in (60, 30):
            bm = np.zeros((h, w), np.uint8)
            pad = max(2, int(0.01 * dpi))
            for wd in words:
                if wd['conf'] >= thr:
                    bm[max(0, int(wd['y0']) - pad):int(wd['y1']) + pad, max(0, int(wd['x0']) - pad):int(wd['x1']) + pad] = 1
            boxes[thr] = bm
        res_ocr = dict(n_words=len(words), n_conf60=sum(1 for wd in words if wd['conf'] >= 60), cpu_ocr=cpu_ocr, ink_in_box60=float(boxes[60][dens > 0.4].mean()) if (dens > 0.4).any() else 0.0)
    t_rm = 0.0
    masks = {}
    for name, PV in VARIANTS.items():
        t = time.process_time()
        mk = F.fold_mask(pasted, dpi, folds, PV)
        rm = F.remove_folds(pasted, dpi, folds, PV, mask=mk)
        t_rm = max(t_rm, time.process_time() - t)
        outs['v' + name] = clean_page(rm, dpi)
        masks[name] = mk
        if name == MAIN and boxes:
            for thr, bm in boxes.items():
                mk2 = F.fold_mask(pasted, dpi, folds, PV, protect=bm)
                outs['vocr%d' % thr] = clean_page(F.remove_folds(pasted, dpi, folds, PV, mask=mk2), dpi)
        if name == MAIN:
            mask, removed = mk, rm
            vc = v0.copy()
            vc[mk > 0] = 255
            outs['vc'] = vc
    # --- metrics
    traj = y0 + np.tan(np.radians(tilt)) * (np.arange(w) - w / 2.0)
    yy = np.arange(h)[:, None]
    dist = np.abs(yy - traj[None, :])
    band = dist <= 0.06 * dpi
    sig = max(0.5, 0.003 * dpi)
    db = cv2.GaussianBlur(dens, (0, 0), sig)
    p95 = np.percentile(db[band], 95)
    core = band & (db >= max(0.04, 0.5 * p95))
    res = dict(idx=idx, pdf=rec['pdf'].split('/')[-1][:30], page=rec['page'], dpi=dpi, strip=strip_name, mode=mode, cls=cls, y=round(y0 / h, 3), tilt=round(tilt, 2),
               ncore=int(core.sum()), det=det, t_det=t_det, t_rm=t_rm)
    if core.sum() < 30:
        res['skip'] = 'no ink in band'
        return res
    de = {k: lab_de_white(v) for k, v in outs.items()}
    for k, v in de.items():
        res['ret_' + k] = float((v[core] >= 6).mean())
    near = cv2.dilate(core.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    for thr, lab in ((10, '10'), (40, '40')):
        foot = (T < 1 - thr / 255.0) & ~near
        res['foot' + lab] = int(foot.sum())
        for k, v in outs.items():
            res[f'surv{lab}_' + k] = int((lum(v)[foot] < 255 - thr).sum())
    extra = band & (lum(ref) >= 250)
    for k, v in outs.items():
        if k != 'ref':
            res['extra_' + k] = int((lum(v)[extra] < 215).sum())
    # nothing outside the band may change (va vs clean of the unpasted removal-free page is not defined; check removal footprint only)
    if boxes:
        res['ocr'] = res_ocr
    res['mask_px'] = int(mask.sum())
    res['removed_changed_px'] = int((removed != pasted).any(2).sum())
    return res


def choose_pages(n_per_class, seed=0):
    rng = np.random.default_rng(seed)
    ids = np.arange(len(index()))
    H = {int(k): v for k, v in CL.items()}
    st = np.array([H[i]['strong'] / (H[i]['H'] * H[i]['W']) for i in ids])
    col = np.array([H[i]['col'] / (H[i]['H'] * H[i]['W']) for i in ids])
    p99 = np.array([H[i]['p99'] for i in ids])
    fr = np.array([H[i]['frac'] for i in ids])
    ok = np.array([H[i]['H'] >= 900 and H[i]['W'] >= 700 for i in ids])
    classes = {'text': ok & (st > 0.01) & (col < 0.001), 'colour': ok & (col > 0.01), 'faint': ok & (p99 < 0.5) & (fr > 0.006)}
    out = []
    for c, m in classes.items():
        sel = rng.choice(ids[m], size=min(n_per_class, int(m.sum())), replace=False)
        out += [(int(i), c) for i in sel]
    return out


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    pages = choose_pages(n)
    jobs = []
    for j, (i, c) in enumerate(pages):
        for mode in ('ink', 'prior'):
            jobs.append((i, STRIPS[(j + (mode == 'prior')) % 4], mode if mode == 'ink' else 'prior', 1000 * j + (mode == 'prior'), c))
    # e2e = detect on the 'prior' instances (customary positions)
    jobs += [(i, s, 'e2e', sd + 7, c) for (i, s, m, sd, c) in jobs if m == 'prior']
    with Pool(12) as p:
        res = p.map(run_instance, jobs, chunksize=1)
    json.dump(res, open(SCR + '/synth_raw.json', 'w'))
    print(len(res), 'instances')
