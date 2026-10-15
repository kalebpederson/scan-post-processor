"""before/after panels: original | B0 | best (hysteresis, flatten) | simple threshold M2 at matched bleed (and keep-mode variants for coloured paper)."""
import sys, json; sys.path.insert(0,'.')
from common import *
from clean import clean_page
from montage import montage
B = lambda: None
METHODS = {
 "B0": lambda rgb, dpi: b0(rgb),
 "BEST": lambda rgb, dpi: clean_page(rgb, dpi),
 "M2thr": lambda rgb, dpi: clean_page(rgb, dpi, gate="none", t_lo=0.08, t_hi=0.14, faint_gain=1.0, gamma=1.0),
}
CASES = [  # name, path, dpi, crop boxes (x,y,w,h native)  -- one 1:1 crop per entry
 ("yellow_paper_image01", str(CORPUS/"image01.jpg"), 300, [(60, 300, 480, 360)]),
 ("pink_paper_bleed_unitreview", str(HERE/"work/pages/390e6aba/pg-002.jpg"), 200, [(1100, 1330, 480, 360), (850, 780, 480, 360)]),
 ("gray_paper_single38", str(CORPUS/"single38.jpg"), 300, [(300, 400, 480, 360)]),
 ("heavy_bleed_complexnum", str(HERE/"work/pages/c26e6e0c/pg-000.jpg"), 300, [(450, 2850, 480, 360)]),
 ("pencil_pair03", str(CORPUS/"pair03_front.jpg"), 300, [(1150, 850, 480, 360), (1100, 1050, 480, 200)]),
 ("pencil_image04_grid", str(CORPUS/"image04.jpg"), 200, [(100, 340, 480, 120), (950, 270, 480, 360)]),
 ("colorcoded_single06", str(CORPUS/"single06.jpg"), 200, [(950, 280, 480, 360)]),
 ("colorcoded_pair01", str(CORPUS/"pair01_front.jpg"), 150, [(300, 420, 480, 360)]),
 ("graphgrid_pair07", str(CORPUS/"pair07_front.jpg"), 150, [(150, 280, 480, 360)]),
 ("epson_notebook_keymap", str(SCANS/"epson_300dpi.png"), 300, [(640, 400, 480, 360), (1450, 450, 480, 360), (1800, 150, 480, 360)]),
]
def tile_row(imgs, caps, width):
    tiles = []
    for im, cap in zip(imgs, caps):
        s = width / im.shape[1]; t = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
        t = cv2.copyMakeBorder(t, 18, 2, 2, 2, cv2.BORDER_CONSTANT, value=(90, 90, 90)); cv2.putText(t, cap, (4, 13), 0, .45, (255, 255, 0), 1); tiles.append(t)
    H = max(t.shape[0] for t in tiles)
    return np.concatenate([cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(90, 90, 90)) for t in tiles], 1)
def run(only=None):
    for name, path, dpi, crops in CASES:
        if only and name not in only: continue
        rgb = load_rgb(path)
        outs = {k: f(rgb, dpi) for k, f in METHODS.items()}
        extra = {}
        if name in ("yellow_paper_image01", "pink_paper_bleed_unitreview", "gray_paper_single38"):
            extra["BEST-keep"] = clean_page(rgb, dpi, paper="keep")
        order = [("original", rgb)] + [(k, v) for k, v in outs.items()] + list(extra.items())
        ncol = len(order); w = 1990 // ncol
        save_rgb(HERE / "panels" / f"{name}_full.png", tile_row([o for _, o in order], [k for k, _ in order], w - 4))
        for i, (x, y, cw, ch) in enumerate(crops):
            cw = min(cw, 1990 // ncol - 4)
            cr = [o[y:y + ch, x:x + cw] for _, o in order]
            save_rgb(HERE / "panels" / f"{name}_crop{i}_1to1.png", tile_row(cr, [f"{k} @1:1 ({x},{y})" for k, _ in order], cw))
if __name__ == "__main__":
    run(sys.argv[1:] or None)
