# R2: cleaning algorithm study (paper white, black text, vivid colour, faint content kept, bleed removed)

> Figures referenced below are not published (they show private scans).

All code and outputs are in `research/r2_cleaning/`. Native resolution throughout. `samples/`, `research/corpus/`, `research/scans/` were only read.
Reproduce: `make_labels.py` -> `pairs2.py` -> `pairs3.py` -> `sweep3..9.py` / `final_eval.py` -> `make_tables.py`, `plot_tradeoff.py`, `panels.py`, `synth.py`, `timing.py`.
Sweeps ran on a 16-page subset (all pages that contain labelled regions + 3 coloured-paper pages) plus all 35 bleed instances; the winner and the reference methods were then re-run on all 56 corpus images + Epson + the 35 bleed fronts (`final_eval.py`, table below).

## Chosen method (`clean.py: clean_page(rgb, dpi, paper="flatten"|"keep", **params)`)
1. Flat-field: per-channel background on a ~50 dpi copy (small opening to remove JPEG overshoot specks, 0.35 in max filter, guard so big dark blobs are not paper, blur). Divide: paper -> 1.0 for any paper colour / shading.
2. Ink density per channel `r_c = max(0, 1 - rgb_c/bg_c)`, `d = max_c r_c` (also fires for pale highlighter).
3. Soft gate `alpha = smoothstep((u - t_lo)/(t_hi - t_lo))`, `u = blur(d) + 0.5*|grad blur(d)|` (sharpness bonus), `t_lo=0.02`, `t_hi=0.14`, raised automatically by a page noise floor (median+3 MAD of paper d; needed for noisy pink/gray paper).
   **Hysteresis**: components of `{d > t_lo}` that touch strong ink (`d > 0.25`) or are large (> 0.5 sq in) keep the low threshold; isolated weak components (bleed ghosts, speckle) must exceed `t_hi_unsup = 0.12`. The ramp is soft, no hard pixel decisions.
4. Tone: `dk_c = clip(r_c/(1-bp) * k)`, `bp = 0.16`, `k` = common gain on all channels up to 3.0 for faint grey ink (d < 0.05 full, fading out by d = 0.25) and not applied to coloured pixels (chroma fade 0.08-0.22). Faint graphite/print is darkened and contrast-boosted with hue preserved; strong ink and highlighter are untouched.
5. `out = 1 - alpha*dk`. `paper="keep"`: multiply by the global paper colour (uniform tint; shading/bleed/noise removed).
All lengths are in inches (dpi-aware). Defaults in `clean.DEFAULTS`.

## Methods compared
| id | description |
|---|---|
| B0 | user baseline `-level 5.098%,86.6667%,0.8822` (numpy re-implementation verified vs magick, max diff 1/255) |
| M1 | flat-field + per-channel levels (window 0.35 in, white clip 0.90, black 0.12); also swept window {0.15,0.35,0.7} x white {0.80..0.95} |
| M2 | flat-field + plain soft threshold on d (no spatial gating), swept t_lo 0.02-0.14, gamma 1/1.5/2 |
| M3 | M2 + sharpness bonus; swept weight 2/4/8 |
| H / HS | hysteresis (+ sharpness bonus, + faint gain, + noise floor); swept t_lo, ramp width, t_hi_unsup, seed, component size, faint gain/ranges |
| L7 | H + long-straight-line support (grids/rules). Tried, rejected: raised bleed survival 2.4x for +0.6 pt retention |
Rows below: `M2_threshold_matched` (t_lo .08) is M2 at about the same bleed survival as FINAL; `M2_threshold_retain` (t_lo .03) is M2 at about the same retention.

## Metric definitions
* **Paper whiteness**: fraction of *pure paper* pixels with all channels >= 250. Pure paper = native ink-density < 0.035 (blurred), >= 0.04 in from any pixel with d > 0.08, inside the page interior (12-96 % x, 3-97 % y; excludes spiral binding). Reported for the 53 white-paper pages and per coloured-paper page (single08 pink, single38 blue-gray, image01 yellow-green). *Background-clean* = same test on all non-ink area (d < 0.15, 0.04 in away from d > 0.15), so it includes bleed, haze and noise.
* **Faint-content retention**: 18 hand-labelled regions in `labels.json` (17 have enough core pixels to be scored): pencil x4, grid x5 (one on pink paper), Epson ruled lines, highlighter x3, light printed grey/dotted text x4, light blue pen x1. The stroke set is the **core** of each region (blurred d >= max(0.04, 0.5 x the region's 95th percentile)), not all pixels with d >= 0.05 (see "Metric check"). Retention = fraction of core pixels with Delta-E to white >= 6 after cleaning (macro average over regions). Contrast = mean Delta-E of core pixels vs white, output vs native.
* **Bleed**: 11 hand-picked bleed-only regions (10 on white paper + 1 on pink): same Delta-E >= 6 survival on pixels with d >= 0.05 (lower is better). Pair-based: bleed region = (verified, registered, mirrored back ink d > 0.3 dilated 0.03 in) AND NOT (front ink d > 0.22 dilated 0.06 in); *survival* = fraction of pixels with native d >= 0.03 inside it that stay non-white (< 250) after cleaning.
* **Ink blackness**: native pixels with max channel <= 90 and Lab chroma < 30; median / p90 of output luminance. Ink-area ratio = count(out luminance < 128)/count(native luminance < 128) shows stroke thickening.
* **Colour**: native pixels with Lab chroma > 25 and d > 0.2 on white-paper pages (coloured-paper pages excluded because the paper itself would count as "coloured"); LCh hue shift (deg) and chroma ratio.
* **Speed**: single-threaded `clean_page`, median of 5, pinned to one core (`taskset`), on a machine whose load average was ~30 (so numbers are conservative).

## Pair verification (bleed ground truth)
The manifest pairs are **not** reliable: none of the 10 pairs (pages 1,2 or 3,4 of each PDF) registered. `pairs2.py` extracted every page of every source PDF (read-only, `pdfimages`), and tested every equal-size page pair: back mirrored horizontally, translation searched +-1.3 in by normalised cross-correlation of the front's *faint layer* (real ink masked out) against the back's ink, both directions, then ECC affine refinement (`pairs3.py`, `pairs_verified.json`). Z-score of the peak against the null distribution of other shifts was 7-30 for true sheets. Result: **18 verified duplex sheets (35 directed front<-back instances)**, typically pages (2,3) or (1,2)/(3,4) of the PDF (the scanner/ADF order differs from the heuristic), shift 0.06-0.29 in horizontally, rotation about 0.01 rad, ECC rho 0.19-0.73; two sheets discarded (Slope/Midpoint 1<-2 and Same-Answer 5<->6, failed refinement). So 18 of the candidate sheets remain; 0 of the manifest's 10 pairs did. Mirror-horizontal was the only orientation that registered.
Bleed in this corpus is very weak (native darkness d 0.05-0.10, desaturated blue/pink), except on pink paper (3.99 Unit Review pp. 3-4), which is the hardest case.

## Metric check (reviewer item 2)
`panels/metric_check_*.png`: the first labelling (all pixels with d >= 0.05) included 2x-wide anti-aliasing skirts and paper; B0 scored 0.42 on it. The core definition hugs the strokes. With it B0 scores 0.66 (pencil 0.89, grid 0.47, light print 0.47). All numbers in this report use the core definition. The faintest grids (image04) are dotted lines with d < 0.05, essentially invisible even in the native image; the core there is mostly intersections.

## Results (all pages; 17 positive regions, 11 negative regions, 35 bleed instances)
| metric | B0 | FINAL_flatten | FINAL_keep | M1_flatfield_levels | M2_threshold_matched | M2_threshold_retain |
|---|---|---|---|---|---|---|
| paper whiteness, 53 white-paper pages (mean / worst page) | 1.0000 / 0.9999 | 1.0000 / 0.9999 | 1.0000 / 0.9999 | 1.0000 / 0.9995 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| background-clean fraction (non-ink area >=250) | 0.966 | 0.996 | 0.967 | 0.998 | 0.999 | 0.996 |
| colored paper single08: whiteness (flatten) | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| colored paper single38: whiteness (flatten) | 0.000 | 0.995 | 0.000 | 0.999 | 0.999 | 0.997 |
| colored paper image01: whiteness (flatten) | 0.847 | 0.992 | 0.992 | 0.999 | 0.998 | 0.995 |
| ink median / p90 luminance (0=black) | 60 / 75 | 38 / 53 | 37 / 53 | 54 / 70 | 48 / 62 | 48 / 62 |
| ink-area ratio (dark px out/in) | 0.98 | 1.30 | 1.33 | 0.97 | 1.15 | 1.15 |
| hue shift mean / p90 (deg) | 3.4 / 8.1 | 2.2 / 4.7 | 2.2 / 4.7 | 2.6 / 5.8 | 1.6 / 3.3 | 1.6 / 3.3 |
| chroma ratio median / p10 | 1.24 / 0.94 | 1.20 / 1.15 | 1.20 / 1.15 | 1.25 / 1.00 | 1.14 / 1.11 | 1.14 / 1.11 |
| faint-content retention (17 regions, core px) | 0.663 | 0.851 | 0.895 | 0.671 | 0.727 | 0.791 |
|   - pencil | 0.89 | 1.00 | 1.00 | 0.93 | 1.00 | 1.00 |
|   - grid | 0.47 | 0.60 | 0.75 | 0.34 | 0.34 | 0.48 |
|   - light_print | 0.47 | 0.83 | 0.83 | 0.51 | 0.57 | 0.69 |
|   - ruled | 0.75 | 1.00 | 1.00 | 0.79 | 0.99 | 0.99 |
|   - highlighter | 0.74 | 1.00 | 1.00 | 0.90 | 0.99 | 1.00 |
|   - light_pen | 0.97 | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 |
| faint contrast: out dE / native dE | 19.3 / 24.5 | 31.5 / 24.5 | 32.8 / 24.5 | 19.6 / 24.5 | 24.1 / 24.5 | 25.4 / 24.5 |
| bleed-only regions survival (11 regions) | 0.098 | 0.016 | 0.107 | 0.029 | 0.020 | 0.205 |
| pair bleed survival, pixel-pooled (35 directed) | 0.319 | 0.039 | 0.352 | 0.016 | 0.007 | 0.091 |
| pair bleed survival, per-instance mean | 0.069 | 0.080 | 0.137 | 0.031 | 0.019 | 0.154 |
| mean darkness in bleed region (native visible-bleed darkness 0.0515) | 0.0068 | 0.0112 | 0.0216 | 0.0016 | 0.0019 | 0.0117 |

Notes: for `FINAL_keep` the alpha/ink maps are identical to flatten; the whiteness-based bleed numbers in that column are not meaningful (the paper is tinted, not white), use the flatten column. keep-mode paper uniformity: mean Delta-E of output paper to the page's true paper colour 0.016 (L std 0.016); image01's whiteness is high only because the "paper" there is mostly the white scanner lid.

## Trade-off curve and knee (`panels/tradeoff_bleed_vs_retention.png`)
Every sweep config is a point (retention vs bleed survival, two views: bleed-only regions, pixel-pooled pair bleed). Plain thresholding (M2, M1) lies on a steep curve: to get retention 0.79 it needs 20 % bleed-region survival, to hold 2 % survival it retains 0.73. Sharpness-only gating (M3) is on or below that curve (it also pulls in bleed edges). Hysteresis + noise floor + faint gain pushes the whole frontier up-left: **retention 0.85 at 1.6 % bleed-region survival and 3.9 % pooled pair survival** (B0: 0.66 at 9.8 % / 31.9 %). Pushing t_lo/ramp further (retention 0.90-0.94) costs 5-18 % bleed survival (steep part of the curve), and the retention gain there is almost all the very faint dotted/grid regions. Knee = bleed-region survival about 2 %, the reviewer's constraint; the chosen point is the highest-retention config below it.
The reviewer's objective is met: retention 0.851 vs B0 0.663; bleed-only survival 0.016 (<= ~2 %); pooled pair bleed 0.039 vs B0 0.319.
A "retain more" preset is `t_hi=0.10` (retention 0.88, bleed 2.6 %, pair 0.15) or `t_lo=0.02, t_hi=0.09 + line_support` (0.90, 8 %, 0.21).

## Speed (single thread, loaded machine)
300 dpi letter page (2544x3295, 8.4 MP): **1.3 s best / 1.8 s median = 155-212 ms/MP**. 200 dpi: 0.7-1.0 s (265 ms/MP incl. cold start variance); 150 dpi (2.1 MP): 0.18-0.26 s (about 120 ms/MP). B0 in numpy: 59-86 ms/MP; `magick -level` including JPEG decode and PNG encode on the same page: 1.5-1.9 s. The cost is dominated by full-resolution float32 planes, the Gaussian/Sobel on the ink map and the connected-components pass. Not yet optimised (gate could run at <=150 dpi and be upsampled).

## Synthetic checks (`synth.py`, `work/synth.json`)
* Yellow highlighter on white paper (multiply (1,1,0.45), (1,1,0.75), pink (1,0.8,0.85) over text bands, flatten): highlighted paper stays visible in 100 % of pixels; Delta-E to white native/output = 68/75, 32/35, 24/31 (it is slightly *more* vivid), text under it stays black (median luminance 2).
* White page tinted yellow (0.95,0.93,0.55) with a 10 % shading gradient: flatten -> paper (255,255,255) in 100 % of paper pixels, L std 0.01; keep -> paper (234,229,135) vs input median (231,226,134), uniform; ink median 2 / 1.3.

## Failure cases
* **Pages dominated by photographs / large grey areas, especially on coloured paper** (single38, art-transformations page): large non-paper regions are treated as paper (guard) and get a complementary colour cast / posterised grey; B0 is no better. Panel `gray_paper_single38_full.png`.
* **Coloured sheet smaller than the scan bed** (image01): the sheet's last ~0.5 in next to the white lid stays tinted under flatten, and `keep` picks the lid white as paper colour.
* **Very faint dotted grids** (image04, grid on pink paper): only 0.25-0.40 retention; they sit at the same darkness as bleed (d 0.05-0.07). Raising retention there raises bleed in proportion.
* **Bleed attached to real ink** (ghost text touching a printed line) is "supported" by hysteresis and survives; mean darkness left in bleed regions is 0.011 vs B0's 0.0068 on white paper (B0's global clip at 221 simply erases everything with d < 0.13, ghost and faint content alike). Per-instance mean survival 0.080 vs B0 0.069 on white paper. FINAL wins on the pooled number because it removes bleed on pink paper, where B0 keeps all of it.
* **Thickening**: the faint-ink gain slightly fattens printed text (ink-area ratio 1.30 vs 0.98 for B0; 1.06 without gain, 1.9 with the first untuned setting). Chosen gain ranges (d 0.05-0.25) are the compromise.
* **JPEG chroma fringes** (yellow/blue dots beside strokes on coloured paper) are in the source and are not removed.
* Left spiral binding / page edges are not specially handled (excluded from metrics).
* Bleed ground truth is weak (d about 0.05-0.10) and the sets are small (11 regions, 18 sheets); retention/bleed differences below ~0.03 are within noise.

## Pre-clipping by Simple Scan (brightness 17, contrast 63)
* 53 of 56 corpus pages have a modal paper value of exactly 255, 77-98 % of interior pixels >= 250. The Epson lossless scan **also** clips: >= 80 % of its paper pixels are 255 (its own auto exposure), so it is not an unprocessed reference, and I cannot prove from it what Simple Scan destroyed.
* What the data does show: the faintest real content sits only 10-25 levels below the clip. Median luminance of the labelled core pixels: image04 grids 235-240 (51 % of core pixels >= 240), yellow highlighter 242, single28 grid 214, vs Epson light print 218 and pencil 200. Content that faint is within the range a +17 brightness shift followed by clipping moves toward 255; any part of it that was already pushed to 255 cannot be recovered by any post-processor, and the flat-field division also loses its information about shading/paper colour (illumination is already clipped away, so M1/M2 only need to cope with page-level colour, not gradients).
* Bleed-through lighter than the clip margin is removed for free by the scanner, which helps; but that is also why bleed here is so faint.
* **Recommendation**: yes, test brightness 0 / contrast 0 (or a mild value) on one handout scanned both ways (front/back, a graph-paper page, a pencil page). If paper then reads ~235-245 with texture and the faint grids/pencil are visibly darker, keep the neutral settings and let this pipeline do the levelling (flatten handles tinted/shaded paper by construction; its noise floor adapts to noisier paper). Expect more bleed-through to be visible at neutral settings; the hysteresis gate and `t_hi_unsup` are the knobs, and thresholds were tuned on clipped data so re-run `final_eval.py` on the new scans. Do not rely on this recommendation until that A/B is done.

## Panels (`panels/`, <= 2000 px wide; columns: original | B0 | BEST (chosen, flatten) | M2thr (plain threshold, matched bleed) [| BEST-keep for coloured paper])
* `yellow_paper_image01_full.png`, `_crop0_1to1.png`: yellow-green paper (flatten and keep).
* `pink_paper_bleed_unitreview_full.png`, `_crop0_1to1.png`, `_crop1_1to1.png`: pink paper, heavy bleed, printed grey grid; flatten and keep.
* `gray_paper_single38_full.png`, `_crop0_1to1.png`: gray/blue paper with photos (failure case).
* `heavy_bleed_complexnum_*`: white paper with bleed ghost.
* `pencil_pair03_*`, `pencil_image04_grid_*`: pencil handwriting, graph grid.
* `colorcoded_single06_*`, `colorcoded_pair01_*`: colour-coded pens.
* `graphgrid_pair07_*`: graph grid.
* `epson_notebook_keymap_full.png`, `_crop0..2_1to1.png`: notebook ruled lines, pencil hatching, halftone keymap text.
* `metric_check_*.png`: stroke mask check. `tradeoff_bleed_vs_retention.png`: the curve.
