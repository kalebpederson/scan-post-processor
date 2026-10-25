"""Punched binding holes (row of >= 6 backing-coloured discs along a border) are painted over, and nothing that merely looks
like a row of dots is (bullets, bubble sheets, dot leaders, glyph columns, a lone circle, coil teeth)."""
import cv2
import numpy as np
import pytest

from scanpp import geometry

from helpers import text_page
from test_flatbed_edge import coil_page

DPI = 100
W, H = 850, 1100
R = 8          # hole radius at 100 ppi (0.17 in diameter)
PITCH = 25


def clean(img):
    return geometry.clean_edges(img, DPI, fill=(255, 255, 255))


def disc(img, cx, cy, r=R, val=(165, 170, 176), rim=(70, 72, 76)):
    cv2.circle(img, (cx, cy), r, val, -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx, cy), (r, r), 0, 200, 340, rim, 2, cv2.LINE_AA)       # darker rim at the top, like the shadow in a hole


def holes_page(side="L", n=40, skip=(), cx=14, y0=24, pitch=PITCH):
    img = text_page(w=W, h=H)
    centres = []
    for k in range(n):
        if k in skip:
            continue
        cy = y0 + k * pitch
        disc(img, cx, cy)
        centres.append((cx, cy))
    if side == "R":
        img = img[:, ::-1].copy()
        # text_page is mirrored too: harmless for this test, only the border matters
    return img, centres


def painted_only_near_holes(img, out, centres, side="L"):
    diff = (img != out).any(axis=2)
    allowed = np.zeros(diff.shape, bool)
    for cx, cy in centres:
        x = cx if side == "L" else W - 1 - cx
        cv2.circle(allowed.view(np.uint8), (x, cy), R + 5, 1, -1)
    return not (diff & ~allowed).any()


@pytest.mark.parametrize("side", ["L", "R"])
def test_hole_row_is_painted(side):
    img, centres = holes_page(side)
    out, info = clean(img)
    assert info.get("holes_" + side) == len(centres)
    for cx, cy in centres:
        x = cx if side == "L" else W - 1 - cx
        assert (out[cy - R:cy + R + 1, x - 3:x + 4] == 255).all()
    assert painted_only_near_holes(img, out, centres, side)


def test_hole_row_top_border():
    img, centres = holes_page("L")
    img = np.ascontiguousarray(np.rot90(img))[:, :]               # row along the top: image is 1100 wide, 850 high
    out, info = geometry.clean_edges(img, DPI, fill=(255, 255, 255))
    assert [k for k in info if k.startswith("holes_")] in (["holes_T"], ["holes_B"])


def test_missing_holes_are_tolerated():
    img, centres = holes_page("L", skip=(3, 4, 17, 30))
    out, info = clean(img)
    assert info["holes_L"] == len(centres)
    assert painted_only_near_holes(img, out, centres)


def test_hole_fused_with_a_mark_only_the_round_part_is_painted():
    img, centres = holes_page("L")
    cy = centres[10][1]
    cv2.line(img, (19, cy - 4), (60, cy - 14), (30, 30, 200), 2, cv2.LINE_AA)       # pen stroke touching the hole
    out, info = clean(img)
    assert info["holes_L"] == len(centres)
    assert (out[cy - R:cy + R + 1, 11:18] == 255).all()
    assert (out[cy - 15:cy - 12, 45:55] == img[cy - 15:cy - 12, 45:55]).all()         # the stroke away from the hole stays


def test_page_end_clipped_hole_is_adopted():
    img, centres = holes_page("L", n=43)             # last hole centred at y = 24 + 42 * 25 = 1074: touches the bottom border
    img2 = img.copy()
    disc(img2, 14, H + 2)
    out, info = clean(img2)
    assert info["holes_L"] >= len(centres)


def test_five_holes_are_not_enough():
    img, _ = holes_page("L", n=5)
    out, info = clean(img)
    assert "holes_L" not in info and np.array_equal(out, reference(img))


def reference(img):
    old = geometry.BINDING_HOLES
    geometry.BINDING_HOLES = False
    try:
        return clean(img)[0]
    finally:
        geometry.BINDING_HOLES = old


def test_switch_restores_previous_output():
    img, _ = holes_page("L")
    assert not np.array_equal(clean(img)[0], reference(img))
    ref = reference(img)
    assert (ref[24 - 5:24 + 5, 10:18] < 250).any()          # previously the holes stayed as gray dots


def _same_as_before(img):
    out, info = clean(img)
    assert not [k for k in info if k.startswith("holes_")]
    assert np.array_equal(out, reference(img))


def column(img, draw, x, n=30, y0=40, pitch=PITCH):
    for k in range(n):
        draw(img, x, y0 + k * pitch)
    return img


def test_printed_bullets_near_the_margin():
    img = text_page(w=W, h=H)
    _same_as_before(column(img, lambda im, x, y: cv2.circle(im, (x, y), 3, (0, 0, 0), -1, cv2.LINE_AA), 14))


def test_large_filled_bullets_would_need_to_be_holes_but_are_black_and_tiny_pitch():
    img = text_page(w=W, h=H)
    _same_as_before(column(img, lambda im, x, y: cv2.circle(im, (x, y), 6, (0, 0, 0), -1, cv2.LINE_AA), 20, pitch=12, n=60))


def test_bubble_sheet_column():
    img = text_page(w=W, h=H)
    _same_as_before(column(img, lambda im, x, y: cv2.circle(im, (x, y), R, (40, 40, 40), 2, cv2.LINE_AA), 40))


def test_dot_leaders():
    img = text_page(w=W, h=H)
    _same_as_before(column(img, lambda im, x, y: cv2.circle(im, (x, y), 1, (0, 0, 0), -1), 20, n=120, pitch=8))


def test_tally_marks_and_dashes():
    img = text_page(w=W, h=H)
    _same_as_before(column(img, lambda im, x, y: cv2.line(im, (x, y - 8), (x, y + 8), (0, 0, 0), 2), 30, n=30))


def test_glyph_column_in_a_table():
    img = text_page(w=W, h=H)
    _same_as_before(column(img, lambda im, x, y: cv2.putText(im, "2", (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.9, (30, 30, 30), 3, cv2.LINE_AA),
                           22, n=30, y0=60, pitch=30))


def test_single_isolated_circle():
    img = text_page(w=W, h=H)
    disc(img, 14, 500)
    _same_as_before(img)


def test_two_columns_of_circles_far_apart_do_not_pair_up():
    img = text_page(w=W, h=H)
    for k in range(3):
        disc(img, 14, 100 + k * 25)
        disc(img, 14, 600 + k * 25)
    _same_as_before(img)


def test_irregular_pitch():
    img = text_page(w=W, h=H)
    rng = np.random.default_rng(1)
    y = 30
    for _ in range(12):
        disc(img, 14, y)
        y += int(rng.integers(21, 55))
    _same_as_before(img)


def test_coil_teeth_are_left_to_the_coil_rule():
    img = coil_page(shading=False)
    out, info = clean(img)
    assert not [k for k in info if k.startswith("holes_")]
    assert np.array_equal(out, reference(img))


def test_text_page_unchanged():
    _same_as_before(text_page(w=W, h=H))
