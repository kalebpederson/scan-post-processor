"""Real crease strips: the trajectory-flattened transmittance (pixel / 255, per channel) of the folds of the test document's fold-only backs
(p4, p6, p14) and the clearest crease of the content page p13 (upper fold, no text within 0.3 in).  Saved to the scratch dir (private-doc derived).
Strip: rows -0.25..+0.25 in around the tracked crease, all columns, 300 ppi; wander_*.npy = the crease's deviation from its straight fit (rows, per column)."""
from common import *
from fold import *
P = FoldParams()
SRC = {'p4': (4, 0.5016, 0.5), 'p6': (6, 0.4994, 1.0), 'p14': (14, 0.336, 0.0), 'p13': (13, 0.339, 0.0)}
if __name__ == '__main__':
    out = {}
    for name, (p, yf, ang) in SRC.items():
        a = td_page(p)
        g = a.astype(np.float32).mean(2)
        dens = ink_density(g, 300.0)
        f = Fold(yf, ang, 0, 0, (0, 0), 0)
        valid = 1.0 - cv2.dilate(((dens > P.strong)).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.float32)
        y0 = yf * a.shape[0]
        tr = track(dens, valid, 300.0, y0, ang, P)
        t = np.tan(np.radians(ang))
        wander = (tr - (y0 + t * (np.arange(a.shape[1]) - a.shape[1] / 2.0))).astype(np.float32)
        np.save(f'{SCR}/wander_{name}.npy', wander)
        J = int(0.25 * 300)
        jj = np.arange(-J, J + 1, dtype=np.float32)
        xs = np.arange(a.shape[1], dtype=np.float32)
        X, Y = np.broadcast_to(xs[None], (len(jj), len(xs))).astype(np.float32), (tr[None, :] + jj[:, None]).astype(np.float32)
        strip = np.stack([cv2.remap(a[..., c].astype(np.float32), X, Y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) for c in range(3)], -1) / 255.0
        np.save(f'{SCR}/strip_{name}.npy', np.minimum(strip, 1.0).astype(np.float32))
        print(name, strip.shape, 'min T %.2f' % strip.min(), 'mean T %.4f' % strip.mean())
