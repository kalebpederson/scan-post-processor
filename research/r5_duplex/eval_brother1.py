import sys, os, json, time, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import duplex
from PIL import Image
import os as _os; from pathlib import Path as _P; _ROOT = str(_P(__file__).resolve().parents[2]); _WORK = _ROOT + '/research/r5_duplex/work'; _os.makedirs(_WORK, exist_ok=True)
BR = _os.environ.get('SCANPP_BROTHER_RAW', _WORK + '/brother/raw') + '/'
ld = lambda i: np.ascontiguousarray(np.array(Image.open(f'{BR}page{i:03d}.tif').convert('RGB')))
A, B = ld(1), ld(2)
mA = duplex.ink_maps(A, 300.); mB = duplex.ink_maps(B, 300.)
f = duplex.register_pair(mA, mB); reg = f['reg']
out = duplex._clean.clean_page(B, 300.)
s, dbg = duplex.bleed_suppression(A, B, reg, debug=True)
new = duplex.apply_suppression(out, s)
np.save('work/br1_out.npy', out); np.save('work/br1_new.npy', new); np.save('work/br1_s.npy', s)
# marker footprint by hue: A's blue / purple marker (Lab hue) warped into B's frame
rA = duplex.source_in_B_frame(A, reg, B.shape[:2])
dA = np.maximum(np.maximum(rA[0], rA[1]), rA[2]); cA = dA - np.minimum(np.minimum(rA[0], rA[1]), rA[2])
# blue/purple: B channel ink density low (blue stays), R,G absorbed:  r_R,r_G >> r_B
bluish = (cA > 0.3) & (rA[0] > rA[2] + 0.15)
bl = cv2.dilate(bluish.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))) > 0
gray = lambda o: o.astype(np.float32) @ np.array([.299, .587, .114], np.float32)
res = {}
for name, o in (('raw', B), ('current', out), ('new', new)):
    dk = 255 - gray(o)
    if name == 'raw':
        # raw darkness relative to local paper: use flat-field d*255
        r = duplex.native_density(B, 300.); dk = np.maximum(np.maximum(r[0], r[1]), r[2]) * 255
    v = dk[bl]
    res[name] = dict(region_px=int(bl.sum()), mean_dark_over_ink=float(v[v > 10].mean()) if (v > 10).any() else 0.0, area_gt40=int((v > 40).sum()), area_gt10=int((v > 10).sum()))
print(json.dumps(res, indent=1))
# notes retention: B's own strong ink outside the (wide) footprint
Sf = dbg['S'] > 0.03
wide = cv2.dilate(Sf.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))) > 0
r = duplex.native_density(B, 300.); dB = np.maximum(np.maximum(r[0], r[1]), r[2])
notes = (dB > 0.25) & ~wide
nw = lambda o: (o.min(-1) < 250)
print('notes px', int(notes.sum()), 'retention current %.4f new %.4f' % (nw(out)[notes].mean(), nw(new)[notes].mean()))
inside = (dB > 0.25) & wide & ~bl
print('strong ink inside wide footprint but not marker-blue: px', int(inside.sum()), 'removed by new:', int((nw(out) & ~nw(new))[inside].sum()))
chg = (out != new).any(-1)
print('changed px', int(chg.sum()), 'changed outside dilated footprint:', int((chg & ~wide).sum()))
# red/green/pencil notes: changed pixels within 3 px of strong non-blue ink
strong = (dB > 0.45)
print('strong-ink (d>.45) px changed:', int((chg & strong).sum()), 'of', int(strong.sum()))
json.dump(res, open('brother1_metrics.json', 'w'), indent=1)
