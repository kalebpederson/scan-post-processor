"""Content-quality regressions: faint text that fades along a line must stay visible word by word (Ace footer), and graphite / black
ink on coloured paper must come out neutral (no mauve pencil, no coloured fringes) without desaturating real pen colours."""
import cv2
import numpy as np
import pytest

from scanpp.clean import clean_page
from test_encoders import needs_render, render, write_pages

SENTENCE = "Fill in the blanks together as a group, then record your answers on Station 6 of your recording sheet."
DPI = 200


def _put(cov, text, org, font, scale, thick, val=1.0):
    """cv2.putText onto a float coverage plane (putText needs uint8): max-merge with `val` * antialiased text."""
    m = np.zeros(cov.shape, np.uint8)
    cv2.putText(m, text, org, font, scale, 255, thick, cv2.LINE_AA)
    np.maximum(cov, m.astype(np.float32) * (val / 255.0), out=cov)


def _line(cov, p0, p1, thick, val=1.0):
    m = np.zeros(cov.shape, np.uint8)
    cv2.line(m, p0, p1, 255, thick, cv2.LINE_AA)
    np.maximum(cov, m.astype(np.float32) * (val / 255.0), out=cov)


def _footer_page(d0=0.27, d1=0.14, seed=1):
    """200 dpi page: a dark body line + a rule + a light-gray italic footer whose darkness fades gently along the line from `d0` to `d1`
    (the measured range of the Ace Stations p6 footer: word peak density 0.27 at the start ... 0.14 at the end (p99 per 100 px column), 255*(1-d) = 186..225)."""
    rng = np.random.default_rng(seed)
    h, w = 700, 1696
    cov = np.zeros((h, w), np.float32)
    boxes = []
    x = 150
    for word in SENTENCE.split():
        (tw, th), _ = cv2.getTextSize(word + " ", cv2.FONT_HERSHEY_TRIPLEX | cv2.FONT_ITALIC, 0.9, 1)
        _put(cov, word, (x, 560), cv2.FONT_HERSHEY_TRIPLEX | cv2.FONT_ITALIC, 0.9, 1)
        boxes.append((word, x - 2, 535, x + tw - 6, 575))
        x += tw
    xs = np.arange(w, dtype=np.float32)
    dens = d0 + (d1 - d0) * np.clip((xs - 150) / (x - 150), 0, 1)
    ink = cv2.GaussianBlur(cov, (0, 0), 1.0) * dens[None, :]
    body = np.zeros((h, w), np.float32)
    _put(body, "Prove: Opposite Sides Theorem", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3, 0.85)
    _line(body, (120, 470), (1580, 470), 3, 0.7)
    g = 255.0 * (1.0 - np.maximum(ink, cv2.GaussianBlur(body, (0, 0), 1.0))) + rng.normal(0, 1.2, (h, w))
    return np.repeat(np.clip(g, 0, 255).astype(np.uint8)[..., None], 3, 2), boxes


def _word_ink(img, box, thr=0.05):
    word, x0, y0, x1, y1 = box
    d = 1.0 - img[y0:y1, x0:x1].min(-1) / 255.0
    return float(np.count_nonzero(d >= thr)), float(d.max())


def test_faint_footer_words_survive_clean():
    img, boxes = _footer_page()
    out = clean_page(img, DPI)
    for box in boxes:
        n_in, _ = _word_ink(img, box, 0.07)
        n_out, peak = _word_ink(out, box, 0.15)
        assert n_out >= 0.5 * n_in and peak >= 0.2, (box[0], n_in, n_out, peak)


def test_old_gate_lost_the_footer_ends():
    """Sanity check of the test itself: without the line support the faint ends of the footer vanish."""
    img, boxes = _footer_page()
    out = clean_page(img, DPI, line_in=0)
    lost = [b[0] for b in boxes if _word_ink(out, b, 0.15)[0] < 0.5 * _word_ink(img, b, 0.07)[0]]
    assert lost, "expected the un-supported gate to lose faint words"


@needs_render
@pytest.mark.parametrize("encoder", ["auto", "mrc", "jpeg"])
def test_faint_footer_words_survive_encode_render(tmp_path, encoder):
    img, boxes = _footer_page()
    out = clean_page(img, DPI)
    write_pages(tmp_path / "f.pdf", [out], dpi=DPI, encoder=encoder)
    got = render(tmp_path / "f.pdf", "pdftoppm", DPI)
    peaks = []
    for box in boxes:
        n_in, _ = _word_ink(img, box, 0.07)
        n_out, peak = _word_ink(got, box, 0.15)
        assert n_out >= 0.4 * n_in and peak >= 0.2, (encoder, box[0], n_in, n_out, peak)
        peaks.append(peak)
    assert min(peaks) >= 0.35 * max(peaks)          # uniformly legible: no word is a ghost next to a bold one


def test_bleed_through_next_to_text_still_vanishes():
    """The line support must not resurrect a faint, smooth ghost sitting on the line of real text."""
    img, boxes = _footer_page(0.5, 0.45)
    g = img[..., 0].astype(np.float32)
    ghost = np.zeros(g.shape, np.float32)
    _put(ghost, "mirrored ghost words", (150, 660), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
    ghost = cv2.GaussianBlur(ghost, (0, 0), 3.0) * 0.06                      # ~6 % density, smooth
    g -= 255 * ghost
    out = clean_page(np.repeat(np.clip(g, 0, 255).astype(np.uint8)[..., None], 3, 2), DPI)
    assert (1.0 - out[600:700].min(-1) / 255.0 >= 0.1).sum() < 20


# ----------------------------------------------------------------------------- colour casts on coloured paper
def _chroma(px):
    px = px.astype(np.int32)
    return px.max(-1) - px.min(-1)


def test_pencil_on_yellow_paper_is_neutral():
    """Graphite is specular: after flat-field division it keeps the paper's tint (measured q = 0.67/0.63/0.64 on yellow paper, which came
    out mauve).  It must be exactly neutral, while a blue pen on the same paper keeps its chroma."""
    rng = np.random.default_rng(0)
    paper = np.array([254, 250, 188], np.float32)
    h, w = 500, 900
    cov = np.zeros((h, w), np.float32)
    _put(cov, "143", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 3, 5)
    _line(cov, (60, 260), (500, 240), 5)
    cov = cv2.GaussianBlur(cov, (0, 0), 1.0)
    pencil_q = np.array([0.66, 0.60, 0.61], np.float32)            # tinted transmittance of graphite on yellow paper
    img = paper[None, None] * (1 - cov[..., None] * (1 - pencil_q)[None, None])
    pen = np.zeros((h, w), np.float32)
    _put(pen, "x=2", (80, 420), cv2.FONT_HERSHEY_SIMPLEX, 3, 5)
    pen = cv2.GaussianBlur(pen, (0, 0), 1.0)
    pen_q = np.array([0.20, 0.30, 0.55], np.float32)    # blue pen
    img = img * (1 - pen[..., None]) + paper[None, None] * pen_q[None, None] * pen[..., None]
    img = np.clip(img + rng.normal(0, 1.0, img.shape), 0, 255).astype(np.uint8)
    out = clean_page(img, 200)
    ink = out[:260].min(-1) < 150                                   # pencil core pixels
    assert ink.sum() > 300
    assert np.percentile(_chroma(out[:260][ink]), 95) <= 6, np.percentile(_chroma(out[:260][ink]), 95)
    base = clean_page(img, 200, neutral_snap=False)
    assert np.median(_chroma(base[:260][ink])) > 8                  # (it was mauve before)
    pink = out[330:].min(-1) < 160
    assert pink.sum() > 100 and np.median(_chroma(out[330:][pink])) > 100      # the blue pen keeps its chroma


def test_black_ink_on_light_blue_paper_has_no_coloured_fringes():
    """Black rules / text on light-blue paper with a scanner black offset and a JPEG-chroma band beside the stroke: ink must come out
    neutral and the band (not ink) must not leave a yellow / green fringe."""
    paper = np.array([180, 236, 255], np.uint8)
    img = np.empty((400, 900, 3), np.uint8)
    img[:] = paper
    for x0 in (50, 450):
        img[150:153, x0:x0 + 350] = (0, 57, 34)                      # black stroke, black-level offset (dark green cast)
        img[153:162, x0:x0 + 350] = (177, 245, 222)                  # chroma-subsampling band under the stroke: yellowish
        img[147:150, x0:x0 + 350] = (140, 190, 210)
    cv2.putText(img, "Is a REFLECTION present?", (60, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 40, 30), 3, cv2.LINE_AA)
    out = clean_page(img, 300)
    assert _chroma(out[148:153, 60:380]).max() <= 12
    band = out[154:162, 60:380]
    assert band.min() >= 235 and _chroma(band).max() <= 12, (band.min(axis=(0, 1)), _chroma(band).max())
    text = out[260:310, 60:700]
    ink = text.min(-1) < 120
    assert ink.sum() > 200 and np.percentile(_chroma(text[ink]), 95) <= 10
    base = clean_page(img, 300, neutral_snap=False)
    assert _chroma(base[154:162, 60:380]).max() > 40                  # the fringe was there before


def test_coloured_pens_keep_their_chroma():
    """Light blue, yellow-green, orange and dark-blue pens on white and on coloured paper are not 'neutral'."""
    pens = {"light blue": (60, 150, 235), "yellow-green": (150, 190, 40), "orange": (240, 130, 30), "dark blue": (30, 40, 130),
            "navy": (75, 77, 115), "burgundy": (112, 68, 90), "dark green": (85, 127, 96)}      # last three: dark pens with chroma/density ~0.25
    for paper in ((255, 255, 255), (254, 250, 188), (255, 200, 220)):
        img = np.empty((300, 7 * 240 + 40, 3), np.uint8)
        img[:] = paper
        x = 30
        for name, c in pens.items():
            cov = np.zeros((300, 7 * 240 + 40), np.float32)
            _put(cov, "ab12", (x, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 6)
            cov = cv2.GaussianBlur(cov, (0, 0), 1.0)
            col = np.array(paper, np.float32) * np.array(c, np.float32) / 255.0
            img = (img * (1 - cov[..., None]) + col[None, None] * cov[..., None]).astype(np.uint8)
            x += 240
        out = clean_page(img, 200)
        ref = clean_page(img, 200, neutral_snap=False)
        for i, name in enumerate(pens):
            sl = (slice(60, 190), slice(30 + 240 * i, 30 + 240 * i + 220))
            core = (255 - ref[sl].min(-1)) > 90
            assert core.sum() > 100, (paper, name)
            c_new, c_ref = np.median(_chroma(out[sl][core])), np.median(_chroma(ref[sl][core]))
            assert c_new >= 0.9 * c_ref, (paper, name, c_new, c_ref)


# ----------------------------------------------------------------------------- robustness of clean_page on small / odd inputs
@pytest.mark.parametrize("paper", ["flatten", "keep"])
def test_clean_page_survives_small_and_odd_sizes(paper):
    """Sizes 16-400 px at 72-600 dpi, text, sparse specks and noise (a lone speck once gave a 1-element index array: OpenCV treats it as
    a scalar -> shape mismatch in the descreened-peak block; centroid NaNs of empty labels crashed the line support)."""
    rng = np.random.default_rng(5)
    for n in (16, 31, 32, 33, 64, 100, 200, 400):
        for dpi in (72, 150, 300, 600):
            for kind in range(3):
                img = np.full((n, n + 7, 3), 250, np.uint8)
                if kind == 0:
                    cv2.putText(img, "Ab", (2, n // 2), cv2.FONT_HERSHEY_SIMPLEX, max(0.3, n / 60), (60, 60, 60), max(1, n // 80), cv2.LINE_AA)
                elif kind == 1:
                    for _ in range(3):
                        x, y = rng.integers(2, n - 2), rng.integers(2, n - 2)
                        v = int(rng.integers(150, 235))
                        cv2.circle(img, (int(x), int(y)), int(rng.integers(0, 3)), (v, v, v), -1)
                else:
                    img = np.clip(img + rng.normal(0, 20, img.shape), 0, 255).astype(np.uint8)
                out = clean_page(img, dpi, paper=paper)
                assert out.shape == img.shape and out.dtype == np.uint8, (n, dpi, kind)


def test_clean_page_400px_at_300dpi_specks():
    rng = np.random.default_rng(1)
    for _ in range(60):
        img = np.full((400, 400, 3), 255, np.uint8)
        for _k in range(int(rng.integers(1, 4))):
            x, y = rng.integers(20, 380, 2)
            v = int(rng.integers(150, 235))
            cv2.circle(img, (int(x), int(y)), int(rng.integers(0, 3)), (v, v, v), -1)
        img = np.clip(img.astype(int) + rng.normal(0, rng.choice([0, 2, 5]), img.shape), 0, 255).astype(np.uint8)
        assert clean_page(img, 300).shape == img.shape
