"""Synthetic checks: (a) yellow highlighter on white paper must survive flatten; (b) yellow-tinted paper with shading flattens to white / keeps tint."""
import sys, json; sys.path.insert(0,'.')
from common import *
from clean import clean_page
from evaluate import dE_white
def run(params, out_json=None):
    rgb = load_rgb(CORPUS/'pair03_front.jpg'); dpi = 300; h, w = rgb.shape[:2]
    res = {}
    # (a) highlighter: translucent yellow multiply over two text bands (strong + light highlighter)
    hl = rgb.astype(np.float32) / 255
    mask = np.zeros((h, w), np.float32)
    bands = [(700, 780, 300, 1500, (1.0, 1.0, 0.45)), (780, 860, 300, 1500, (1.0, 1.0, 0.75)), (1000, 1080, 300, 1500, (1.0, 0.8, 0.85))]
    ink = (rgb.min(-1) < 120)
    outs = []
    for (y0, y1, x0, x1, col) in bands:
        m = np.zeros((h, w), np.float32); m[y0:y1, x0:x1] = 1; m = cv2.GaussianBlur(m, (0, 0), 2)
        hl = hl * (1 - m[..., None] * (1 - np.array(col, np.float32)[None, None, :]))
    hl8 = np.clip(hl * 255 + .5, 0, 255).astype(np.uint8)
    o = clean_page(hl8, dpi, paper="flatten", **params)
    for (y0, y1, x0, x1, col), nm in zip(bands, ("yellow_strong", "yellow_light", "pink_light")):
        reg = np.zeros((h, w), bool); reg[y0 + 8:y1 - 8, x0 + 8:x1 - 8] = True; reg &= ~(cv2.dilate(ink.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
        res["hl_" + nm] = dict(native_dE=float(dE_white(hl8[reg][::50]).mean()), out_dE=float(dE_white(o[reg][::50]).mean()), frac_visible=float((dE_white(o[reg][::50]) >= 6).mean()))
    inkreg = ink & (rgb.min(-1) < 80)
    res["text_under_highlight_out_lum_med"] = float(np.median(o[inkreg & (np.arange(h)[:, None] > 700) & (np.arange(h)[:, None] < 860)].mean(-1)))
    save_rgb(WORK / 'synth_hl.png', np.concatenate([hl8[600:1150, 250:1600], o[600:1150, 250:1600]], 0))
    # (b) yellow paper w/ shading
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shade = 0.90 + 0.10 * (xx / w * 0.5 + yy / h * 0.5)
    tint = np.array([0.95, 0.93, 0.55], np.float32)
    yp = np.clip(rgb.astype(np.float32) / 255 * tint * shade[..., None] * 255 + .5, 0, 255).astype(np.uint8)
    for pm in ("flatten", "keep"):
        o2 = clean_page(yp, dpi, paper=pm, **params)
        d, _ = darkness(rgb, dpi); ds = cv2.GaussianBlur(d, (0, 0), 1.8)
        P = (ds < 0.035) & ~(cv2.dilate((ds > 0.08).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0); P[:int(.03 * h)] = 0; P[:, :int(.12 * w)] = 0
        res[f"yellow_paper_{pm}"] = dict(white_frac=float((o2.min(-1) >= 250)[P].mean()), paper_out_rgb_med=np.median(o2[P], 0).tolist(), paper_in_rgb_med=np.median(yp[P], 0).tolist(),
                                          paper_out_L_std=float(o2[P].mean(-1).std()), ink_out_med=float(np.median(o2[(rgb.min(-1) < 80)].mean(-1))))
        save_rgb(WORK / f'synth_yp_{pm}.png', o2)
    save_rgb(WORK / 'synth_yp_in.png', yp)
    return res
if __name__ == "__main__":
    print(json.dumps(run({}), indent=1))
