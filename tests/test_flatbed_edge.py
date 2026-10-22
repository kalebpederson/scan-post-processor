"""Flatbed scans of a sheet shorter than the glass: the detached paper-edge line (and the coil-end fringe) are painted over,
and content near the edge (rules, footers, table borders, handwriting, highlighter) is not."""
import numpy as np
import pytest

from scanpp import boundary, geometry

from helpers import text_page

DPI = 100
W, H = 850, 1100


def flatbed(line_y=1076, line=(222, 222, 226), thick=2, tilt_px=3, strip=True, strip_h=5, content=None):
    """White page with text, a grey paper-edge line `line_y` rows from the top (tilted by `tilt_px` over the width), white
    scanner lid beyond it and (optionally) the dark scanner frame at the border."""
    img = text_page(w=W, h=H)
    img[line_y - 4:] = 255                                     # lid (white) beyond the paper's edge
    xs = np.arange(W)
    ys = (line_y + np.round(tilt_px * xs / W)).astype(int)
    for dy in range(thick):
        img[ys + dy, xs] = line
    if strip:
        img[H - strip_h:] = (35, 35, 38)
        img[H - strip_h - 1] = (140, 140, 142)
    if content:
        content(img)
    return img


def clean(img):
    return geometry.clean_edges(img, DPI, fill=(255, 255, 255))


def reference(img, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(boundary, "EDGE_LINE", False)
        return geometry.clean_edges(img, DPI, fill=(255, 255, 255))[0]


def test_edge_line_removed_with_lid_and_frame():
    img = flatbed()
    out, info = clean(img)
    assert "boundary_B" in info
    assert (out[1070:] == 255).all()                           # line, lid and frame are paper now
    assert np.array_equal(out[:1060], img[:1060])              # nothing above the line was touched


def test_edge_line_removed_when_faint_and_thick_and_tilted():
    img = flatbed(line=(238, 238, 240), thick=3, tilt_px=6)
    out, info = clean(img)
    assert "boundary_B" in info
    assert (out[1070:] == 255).all()


@pytest.mark.parametrize("variant", ["rule_02", "rule_035", "rule_05", "footer_line_with_strip", "table_border", "handwriting",
                                     "thick_black_rule", "short_rule", "text_on_lid", "no_strip_line"])
def test_edge_line_does_not_touch_content(variant, monkeypatch):
    def rule(y, x0=0, x1=W, val=90, th=2):
        def f(img):
            img[y:y + th, x0:x1] = val
        return f

    if variant == "rule_02":                                    # printed rule 0.2 in from the bottom, blank paper below, frame at the border
        img = flatbed(line_y=1078, line=(255, 255, 255), content=rule(H - 20))
    elif variant == "rule_035":
        img = flatbed(line_y=1078, line=(255, 255, 255), content=rule(H - 35))
    elif variant == "rule_05":
        img = flatbed(line_y=1078, line=(255, 255, 255), content=rule(H - 50, val=200))
    elif variant == "footer_line_with_strip":                   # a margin-to-margin footer rule: does not reach the page edges
        img = flatbed(line_y=1078, line=(255, 255, 255), content=rule(H - 30, 60, W - 60, val=232))
    elif variant == "table_border":
        def tb(img):
            img[H - 45:H - 43, 20:W - 20] = 120
            img[H - 200:H - 43, 20:22] = 120
            img[H - 200:H - 43, W - 22:W - 20] = 120
        img = flatbed(line_y=1078, line=(255, 255, 255), content=tb)
    elif variant == "handwriting":
        def hw(img):
            xs = np.arange(60, W - 60)
            ys = (H - 24 + 5 * np.sin(xs / 17.0)).astype(int)
            for dy in (0, 1, 2):
                img[ys + dy, xs] = (30, 30, 120)
        img = flatbed(line_y=1078, line=(255, 255, 255), content=hw)
    elif variant == "thick_black_rule":                         # full-width but black and 4 px thick: printed, not a sheet edge
        img = flatbed(line_y=1078, line=(255, 255, 255), content=rule(H - 26, val=15, th=4))
    elif variant == "short_rule":
        img = flatbed(line_y=1078, line=(255, 255, 255), content=rule(H - 26, 300, 500, val=232))
    elif variant == "text_on_lid":                              # edge line, but something is written on the 'lid' beyond it
        import cv2
        def txt(img):
            cv2.putText(img, "note", (400, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        img = flatbed(strip=False, content=txt)
    elif variant == "no_strip_line":                            # a faint full-width line near the bottom but no scanner frame
        img = flatbed(strip=False)
    out, info = clean(img)
    ref = reference(img, monkeypatch)
    assert not [k for k in info if k.startswith("boundary_")], info
    assert np.array_equal(out, ref)                              # identical to the behaviour without the rule
    # and the printed content is still there
    changed = (out != img).any(axis=2)
    if variant in {"rule_02", "rule_035", "rule_05", "footer_line_with_strip", "table_border", "thick_black_rule", "short_rule"}:
        assert not changed[:H - 60].any()


def coil_page(shading=True):
    """Coil on the right edge (teeth every 0.25 in, 0.3 in deep) and a smooth lilac shadow patch where the coil ends."""
    img = text_page(w=W, h=H)
    img[:, W - 6:] = (30, 30, 30)
    for y in range(20, H - 20, 25):
        img[y:y + 9, W - 32:] = (25, 25, 28)
    if shading:
        img[0:60, W - 62:W - 34] = (226, 222, 233)
    return img


def test_coil_fringe_is_painted_but_handwriting_is_not(monkeypatch):
    img = coil_page()
    img[300:304, W - 140:W - 36] = (20, 20, 130)               # a thin pen stroke running into the coil band
    img[200:230, W - 62:W - 36] = (255, 250, 120)              # a highlighter patch next to the band
    out, info = clean(img)
    assert info.get("coil_R", 0) > 0.2
    assert info["fringe_px"] > 0
    assert (out[2:55, W - 60:W - 40] == 255).all()              # shadow patch at the coil end is gone
    assert np.array_equal(out[300:304, W - 140:W - 50], img[300:304, W - 140:W - 50])    # handwriting untouched
    assert np.array_equal(out[205:225, W - 60:W - 40], img[205:225, W - 60:W - 40])      # saturated patch untouched
    with monkeypatch.context() as m:
        m.setattr(geometry, "FLATBED_COIL_FRINGE", False)
        ref, _ = clean(img)
    assert not np.array_equal(out, ref)


def test_coil_fringe_is_a_noop_without_a_shadow(monkeypatch):
    img = coil_page(shading=False)
    out, _ = clean(img)
    with monkeypatch.context() as m:
        m.setattr(geometry, "FLATBED_COIL_FRINGE", False)
        ref, _ = clean(img)
    assert np.array_equal(out, ref)


def corner_page(strip_w):
    """Dark strip down the left border and a dark bar along the bottom: the two strip runs meet in the corner, where each
    is too long for the other's painting, so the corner block is left behind by the strip logic alone."""
    img = text_page(w=W, h=H)
    img[:, :strip_w] = (60, 60, 70)
    img[-7:, :] = (40, 40, 45)
    return img


def test_corner_dot_is_painted(monkeypatch):
    img = corner_page(7)
    out, info = clean(img)
    assert info.get("corner_dots", 0) >= 1
    assert (out[-7:, :7] == 255).all()
    with monkeypatch.context() as m:
        m.setattr(geometry, "FLATBED_CORNER_DOTS", False)
        ref, _ = clean(img)
    assert not (ref[-7:, :7] == 255).all()                    # the previous behaviour left it
    assert np.array_equal(out[:H - 20, 30:], ref[:H - 20, 30:])  # and nothing else differs


def reference_corner(img, monkeypatch):
    with monkeypatch.context() as m:
        m.setattr(geometry, "FLATBED_CORNER_DOTS", False)
        return clean(img)[0]


def test_corner_dot_needs_a_painted_strip_next_to_it(monkeypatch):
    img = text_page(w=W, h=H)
    img[-10:, :10] = (40, 40, 45)                               # a dark speck in the corner, no scanner strip at all
    out, info = clean(img)
    assert "corner_dots" not in info
    assert np.array_equal(out, reference_corner(img, monkeypatch))


def test_border_hairline_only_on_coil_pages(monkeypatch):
    coil = coil_page(shading=False)
    coil[-1:, :] = (205, 205, 210)                              # light frame penumbra along the whole bottom border
    out, info = clean(coil)
    assert info.get("hairline_B", 0) >= 1
    assert (out[-1:, 20:W - 40] == 255).all()
    plain = text_page(w=W, h=H)
    plain[-1:, :] = (205, 205, 210)                             # the same line on a page without a coil is left to the old logic
    out2, info2 = clean(plain)
    assert not [k for k in info2 if k.startswith("hairline_")]
    with monkeypatch.context() as m:
        m.setattr(geometry, "FLATBED_HAIRLINE", False)
        ref, _ = clean(coil)
    assert not np.array_equal(out, ref)


def test_border_hairline_ignores_partial_borders():
    coil = coil_page(shading=False)
    coil[-1:, 100:300] = (205, 205, 210)                        # a short mark at the border, not a frame line
    out, info = clean(coil)
    assert not [k for k in info if k.startswith("hairline_")]
