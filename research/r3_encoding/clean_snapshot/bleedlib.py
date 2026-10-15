"""Bleed-region construction for a verified directed pair."""
import sys, json; sys.path.insert(0, '.')
from common import *

def accepted_pairs(path=HERE / "pairs_verified.json"):
    P = json.load(open(path))
    return [p for p in P if p["ncc_after"] >= 0.18 and (p["ecc_rho"] or 0) >= 0.18 and max(abs(p["shift_in"][0]), abs(p["shift_in"][1])) < 0.4]

def bleed_masks(p, rgb_front=None, d_front=None, back_ink_thr=0.30, front_ink_thr=0.22, dil_back_in=0.03, dil_front_in=0.06):
    """returns dict: region (bool HxW: back ink dilated & not front ink dilated), visible (region & native d_front>=0.03), d_front, front_ink"""
    dpi = p["dpi"]
    rf = load_rgb(p["front_file"]) if rgb_front is None else rgb_front
    h, w = rf.shape[:2]
    if d_front is None: d_front, _ = darkness(rf, dpi)
    rb = load_rgb(p["back_file"])[:, ::-1]
    db, _ = darkness(np.ascontiguousarray(rb), dpi)
    if db.shape != d_front.shape: db = cv2.resize(db, (w, h))
    W = np.array(p["warp_full"], np.float32)
    ink_b = (db > back_ink_thr).astype(np.uint8) * 255
    ink_b = cv2.warpAffine(ink_b, W, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP) > 127
    k = lambda r_in: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(1, int(round(r_in * dpi))) + 1,) * 2)
    back_d = cv2.dilate(ink_b.astype(np.uint8), k(dil_back_in)) > 0
    front_ink = d_front > front_ink_thr
    front_d = cv2.dilate(front_ink.astype(np.uint8), k(dil_front_in)) > 0
    valid = np.zeros((h, w), bool); valid[int(.03 * h):int(.97 * h), int(.12 * w):int(.96 * w)] = True
    region = back_d & ~front_d & valid
    visible = region & (d_front >= 0.03)
    return dict(region=region, visible=visible, d_front=d_front, front_ink=front_ink, back_ink=ink_b)
