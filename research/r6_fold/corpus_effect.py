"""Effect of the fold pipeline on the corpus pages where the detector fires (942, 943, 1042 = creased pink paper; 1901, 1902 = folded booklet edge):
pixels changed in the cleaned output, inside / outside +-0.1 in of the fold rows; panel research/r6_fold/panels/corpus_detections.png (raw | shipped | with removal)."""
import json
import numpy as np, cv2
from common import *
import fold as F
from scanpp import geometry
from scanpp.clean import clean_page
P = F.FoldParams()
res = {}
tiles = []
for k in (942, 943, 1042, 1901, 1902):
    a, rec = native_page(k)
    dpi = float(rec['dpi'])
    a = geometry._clean_edges_g(np.array(a), dpi, inplace=True)[0]
    folds = F.detect_folds(a, dpi, P)
    cur = clean_page(a, dpi)
    new = clean_page(F.remove_folds(a, dpi, folds, P), dpi)
    h = a.shape[0]
    keep = np.ones(h, bool)
    for f in folds:
        keep[max(0, int(f.y_frac * h - 0.1 * dpi)):int(f.y_frac * h + 0.1 * dpi)] = False
    d = (cur != new).any(2)
    dd = np.abs(cur.astype(int) - new.astype(int)).max(2)
    res[k] = dict(pdf=rec['pdf'].split('/')[-1][:40], page=rec['page'], folds=[(round(f.y_frac, 3), f.info['tier']) for f in folds], changed_px=int(d.sum()),
                  changed_px_outside_bands=int(d[keep].sum()), max_abs_diff_outside=int(dd[keep].max()))
    print(k, res[k])
    y = int(folds[0].y_frac * h)
    x1 = min(a.shape[1], int(1500 * dpi / 200))
    rows = []
    for im, t in ((a, 'raw'), (cur, 'shipped cleaner'), (new, 'with fold removal (%d px changed)' % int(d.sum()))):
        c = im[y - int(0.35 * dpi):y + int(0.35 * dpi), :x1].copy()
        c = cv2.resize(c, (1000, int(c.shape[0] * 1000 / c.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.rectangle(c, (0, 0), (8 + 7 * len('%d %s' % (k, t)), 16), (255, 255, 255), -1)
        cv2.putText(c, '%d %s' % (k, t), (3, 12), 0, 0.42, (0, 0, 200), 1)
        rows.append(c)
    tiles.append(np.vstack(rows + [np.full((4, 1000, 3), 128, np.uint8)]))
cv2.imwrite(ROOT + '/research/r6_fold/panels/corpus_detections.png', np.vstack(tiles)[:, :, ::-1])
json.dump(res, open('corpus_effect.json', 'w'), indent=1)
