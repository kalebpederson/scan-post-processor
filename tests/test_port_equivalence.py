"""The package must reproduce the research implementations, except where it intentionally diverges.

clean_page: the cleaner now INTENTIONALLY diverges from research/r2_cleaning/clean.py for light halftoned / thin print (peak
support + gate floor on a lightly descreened density + descreened tone in light regions; see clean.DEFAULTS pk_* / ht_*).
  * with the halftone handling switched off (LEGACY) it must still reproduce the research version (max abs diff <= 1:
    the speed-optimised clean_page reformulates the maths as transmittance q = rgb/bg and only evaluates the tone curve where
    the gate is open) -> guards the shared core;
  * with the defaults the output must stay close to the research version on ordinary pages (bounded mean / tail diff and
    non-white area) and match the unoptimised clean_page_reference; the R2 quality metrics are pinned by
    `bench/eval_clean.py` (research baseline: retention 0.851, bleed-only 0.016, pooled pair bleed 0.039; with the halftone fix
    0.876 / 0.016 / 0.042) and the halftone behaviour by tests/test_halftone.py, a golden aggregate below and bench/halftone/.
edges / deskew: bit-identical or max-abs-diff <= 1 (edge painting is exact).

Skew and blank score were speed-optimised after the port (see bench/ and the commit message for the gates):
  * skew: projection voter uses a sparse row-profile + parabolic coarse-to-fine search, LSD runs at 0.75x work
    resolution, ink_map smooths at half resolution.  On the R4 synthetic ground truth the ensemble error is unchanged
    (mean 0.1177 -> 0.1163 deg, p95 0.318 -> 0.315 deg, same failure rate), so estimates are only required to agree when
    both are reliable (conf >= MIN_CONF), and the deskew decision must be identical;
  * blank score: ink_map smoothing at half resolution moves the score by ~0.02 % (relative); the blank decision must be
    identical (on the labelled R4 set the nearest non-blank page scores 232 vs the threshold 40, no false blanks)."""
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from scanpp import clean as new_clean
from scanpp import geometry as new_geo

RESEARCH = Path(__file__).resolve().parent.parent / "research"
CLEAN_PY = RESEARCH / "r2_cleaning" / "clean.py"
GEO_PY = RESEARCH / "r4_geometry" / "geometry.py"

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not (CLEAN_PY.exists() and GEO_PY.exists()), reason="research/ not present")]

IMAGES = ["pair01_front.jpg", "image02.jpg", "single05.jpg"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def old():
    return _load("old_clean", CLEAN_PY), _load("old_geo", GEO_PY)


@pytest.fixture(scope="module", params=IMAGES)
def img(request, corpus_dir):
    p = corpus_dir / request.param
    if not p.exists():
        pytest.skip(f"{p} missing")
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


LEGACY = dict(desc_sigma_in=0, line_in=0, neutral_snap=False)     # halftone handling, line support and neutral-ink snap off: the research cleaner


def _close(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 1


def test_clean_page_core_unchanged_when_halftone_off(old, img):
    oc, _ = old
    for paper in ("flatten", "keep"):
        _close(new_clean.clean_page(img, 150, paper=paper, **LEGACY), oc.clean_page(img, 150, paper=paper))


def test_clean_page_defaults_close_to_research(old, img):
    oc, _ = old
    for paper in ("flatten", "keep"):
        a = new_clean.clean_page(img, 150, paper=paper)
        b = oc.clean_page(img, 150, paper=paper)
        d = np.abs(a.astype(int) - b.astype(int))
        assert d.mean() < 1.0 and (d > 8).mean() < 0.03           # measured: mean 0.08-0.34, tail 0.3-1.2 %
        nw_a, nw_b = (a.min(-1) < 250).mean(), (b.min(-1) < 250).mean()
        assert 0.97 * nw_b <= nw_a <= 1.10 * nw_b                  # ink area changes by < 10 % (halftone/thin light print only)
        assert np.abs(a.astype(int) - new_clean.clean_page_reference(img, 150, paper=paper).astype(int)).mean() < 0.05


def test_clean_page_golden_synthetic_halftone():
    """Frozen aggregate of the cleaned synthetic 150 lpi light-gray halftone text page (deterministic input; the line support added 1.3 % density)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_halftone import _page
    rgb, _ = _page(150, 0.30, 1.2, 2, 1.0)
    d = 255 - new_clean.clean_page(rgb, 300).min(-1).astype(int)
    assert abs(int(d.sum()) - 3639288) < 0.01 * 3639288
    assert abs(int((d >= 64).sum()) - 35852) < 0.02 * 35852


def test_skew_and_deskew(old, img):
    _, og = old
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    a1, c1 = new_geo.estimate_skew(gray, 150)
    a0, c0 = og.estimate_skew(gray, 150)
    if c0 >= new_geo.MIN_CONF and c1 >= new_geo.MIN_CONF:
        assert abs(a1 - a0) < 0.1 and abs(c1 - c0) < 0.35
    assert new_geo.should_deskew(a1, c1) == og.should_deskew(a0, c0)
    _close(new_geo.deskew(img, 1.3, (255, 255, 255)), og.deskew(img, 1.3, (255, 255, 255)))
    _close(new_geo.deskew(img, 1.3), og.deskew(img, 1.3))


def test_edges_and_blank(old, img, monkeypatch):
    # the R4 prototype predates the flatbed edge rules (research/r7_flatbed) and the binding-hole rule (research/r9_holes): compare with them switched off
    from scanpp import boundary
    monkeypatch.setattr(boundary, "EDGE_LINE", False)
    for k in ("FLATBED_CORNER_DOTS", "FLATBED_HAIRLINE", "FLATBED_COIL_FRINGE", "BINDING_HOLES"):
        monkeypatch.setattr(new_geo, k, False)
    _, og = old
    m1, i1 = new_geo.edge_artifact_mask(img, 150)
    m0, i0 = og.edge_artifact_mask(img, 150)
    assert np.array_equal(m1, m0) and i1 == i0
    o1, _ = new_geo.clean_edges(img, 150)
    o0, _ = og.clean_edges(img, 150)
    _close(o1, o0)
    b1, b0 = new_geo.blank_score(img, 150), og.blank_score(img, 150)
    assert abs(b1 - b0) <= 1e-3 * max(b0, 100.0)
    assert new_geo.is_blank(img, 150) == og.is_blank(img, 150)
