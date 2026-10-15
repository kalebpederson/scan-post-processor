"""OCR round-trip legibility metric, copied from research/r1_resolution (lib.py: ocr/norm; epson_eval.py: score; corpus_eval.py: recall).
Differences from R1: the image OCR'd is a *decoded PDF render* (pdftoppm/pdfium at 300 dpi) instead of a down/up-sampled raster."""
import os, re, csv, io, subprocess, tempfile, json, collections
import numpy as np, cv2
from config import R1, TMP, WORK, CORPUS, SCANS, pages, load_src

def norm(tok):
    t = tok.strip()
    if not re.search(r'[A-Za-z0-9]', t): return None
    return re.sub(r'^[^\w]+|[^\w]+$', '', t)

def ocr(g, dpi, psm=11):
    fd, p = tempfile.mkstemp(suffix='.png', dir=TMP); os.close(fd)
    cv2.imwrite(p, g)
    env = dict(os.environ, OMP_THREAD_LIMIT='1')
    out = subprocess.run(['tesseract', p, 'stdout', '-l', 'eng', '--psm', str(psm), '--oem', '1', '--dpi', str(int(dpi)), 'tsv'],
                         capture_output=True, text=True, env=env).stdout
    os.remove(p)
    rows = []
    for r in csv.DictReader(io.StringIO(out), delimiter='\t', quoting=csv.QUOTE_NONE):
        if r['level'] != '5': continue
        t = norm(r['text'] or '')
        if not t: continue
        rows.append(dict(text=t, conf=float(r['conf']), x=int(r['left']), y=int(r['top']), w=int(r['width']), h=int(r['height']),
                         line=(int(r['block_num']), int(r['par_num']), int(r['line_num']))))
    return rows

def recall(ref_words, hyp):
    rc = collections.Counter(ref_words); hc = collections.Counter(w['text'] for w in hyp)
    return sum(min(rc[t], hc[t]) for t in rc) / max(1, len(ref_words))

# ---- corpus reference: native OCR of the ORIGINAL 300-dpi scan, psm 3, conf >= 90 (as R1 C1)
CORPUS_REF_JSON = WORK / "res" / "corpus_ref.json"
def native300_pages():
    return [p for p in pages() if p["kind"] == "corpus" and p["native"] == 300 and p["name"] != "single26"]
def corpus_ref(name):
    if CORPUS_REF_JSON.exists():
        d = json.load(open(CORPUS_REF_JSON))
        if name in d: return d[name]
    return None
def build_corpus_ref():
    from multiprocessing import Pool
    P = native300_pages()
    def f(p): pass
    with Pool(16) as pool: res = pool.map(_ref_job, P)
    d = {p["name"]: r for p, r in zip(P, res)}
    json.dump(d, open(CORPUS_REF_JSON, "w")); return d
def _ref_job(p):
    g = cv2.cvtColor(load_src(p)[:, :, ::-1], cv2.COLOR_BGR2GRAY)
    return [w['text'] for w in ocr(g, 300, 3) if w['conf'] >= 90]

# ---- Epson: R1 consensus reference (600-dpi grid, rotated page), score() copied verbatim from epson_eval.py
REF = json.load(open(R1 / "work" / "consensus.json"))
def _cls(w):
    d = w['dark']; return 'bold' if d >= 50 else ('mid' if d >= 35 else 'light')
for w in REF: w['cls'] = _cls(w); w['bw'] = (w['cls'] == 'bold' and w['text'].isalpha() and len(w['text']) >= 4)

def score(hyp, scale):
    """hyp boxes in image px; scale = 600grid px / image px (positional recall with per-image linear registration)"""
    H = [dict(text=h['text'], cx=(h['x'] + h['w'] / 2) * scale, cy=(h['y'] + h['h'] / 2) * scale) for h in hyp]
    rc0 = collections.Counter(w['text'] for w in REF); hc0 = collections.Counter(h['text'] for h in H)
    pr = []
    for w in REF:
        if rc0[w['text']] == 1 and hc0[w['text']] == 1:
            h = [h for h in H if h['text'] == w['text']][0]; pr.append((h['cx'], h['cy'], w['x'] + w['w'] / 2, w['y'] + w['h'] / 2))
    if len(pr) >= 25:
        pr = np.array(pr); keep = np.ones(len(pr), bool)
        for _ in range(4):
            ax = np.polyfit(pr[keep, 0], pr[keep, 2], 1); ay = np.polyfit(pr[keep, 1], pr[keep, 3], 1)
            res = np.hypot(np.polyval(ax, pr[:, 0]) - pr[:, 2], np.polyval(ay, pr[:, 1]) - pr[:, 3]); keep = res < max(8, 2.5 * np.median(res[keep]))
            if keep.sum() < 15: break
        for h in H: h['cx'] = float(np.polyval(ax, h['cx'])); h['cy'] = float(np.polyval(ay, h['cy']))
    rc = collections.Counter(w['text'] for w in REF); hc = collections.Counter(h['text'] for h in H)
    ms = sum(min(rc[t], hc[t]) for t in rc)
    used = set(); hit = []
    for w in REF:
        cx, cy = w['x'] + w['w'] / 2, w['y'] + w['h'] / 2; best = None
        for j, h in enumerate(H):
            if j in used or h['text'] != w['text']: continue
            d = abs(h['cx'] - cx) + abs(h['cy'] - cy)
            if d < 20 and (best is None or d < best[0]): best = (d, j)
        if best: used.add(best[1]); hit.append(True)
        else: hit.append(False)
    out = dict(n_hyp=len(H), ms_recall=ms / len(REF), pos_recall=float(np.mean(hit)))
    for c in ['bold', 'mid', 'light']:
        idx = [i for i, w in enumerate(REF) if w['cls'] == c]
        out['pos_' + c] = float(np.mean([hit[i] for i in idx])); out['n_' + c] = len(idx)
    hd = [i for i, w in enumerate(REF) if w['bw']]
    out['pos_heading'] = float(np.mean([hit[i] for i in hd]))
    return out

def epson_eval(render_rgb):
    """round trip as R1: viewer-scale (300 dpi render) -> bilinear to the 600 grid -> tesseract psm 11 --dpi 600 -> positional recall"""
    g = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2GRAY)
    up = cv2.resize(g, (7016, 5096), interpolation=cv2.INTER_LINEAR)
    return score(ocr(up, 600, 11), 1.0)

def corpus_eval(name, render_rgb):
    ref = corpus_ref(name)
    g = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2GRAY)
    return dict(recall=recall(ref, ocr(g, 300, 3)), nref=len(ref))
