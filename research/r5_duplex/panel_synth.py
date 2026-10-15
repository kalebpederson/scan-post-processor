import glob, numpy as np, cv2
from PIL import Image
fs = sorted(glob.glob('work/v/synthex_*.png'))
pick = [f for f in fs if 'synthex_blue' in f][:3] + [f for f in fs if 'synthex_neutral' in f][:2] + [f for f in fs if 'synthex_red' in f][:1]
rows = []
for f in pick:
    im = np.array(Image.open(f)); h, w = im.shape[:2]; w3 = w // 3
    hdr = np.full((18, w, 3), 255, np.uint8)
    cv2.putText(hdr, f.split('synthex_')[1][:-4] + ':  composite (raw) | current scanpp | with duplex pass', (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 0, 0), 1, cv2.LINE_AA)
    rows.append(np.vstack([hdr, im]))
W = max(r.shape[1] for r in rows)
rows = [np.pad(r, ((0, 4), (0, W - r.shape[1]), (0, 0)), constant_values=200) for r in rows]
Image.fromarray(np.vstack(rows)).save('panels/synthetic_overlap.png'); print(np.vstack(rows).shape)
