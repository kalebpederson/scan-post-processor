# R4 — Deskew, blank-page detection, edge artifacts

> Figures referenced below are not published (they show private scans).

(Transcribed by the orchestrator from the R4 agent's final report; the agent could not write files at the end of its run.)

Data: all 1969 scan pages in `samples/` (291 scan PDFs), decoded losslessly, cached at 100 ppi (`cache/`, rebuildable via `extract.py`). Timings are single-core CPU on a loaded machine (upper bounds).

Angle convention: positive = content rotated counter-clockwise on screen (baseline rising to the right). `deskew(img, angle)` rotates by −angle.

## Summary

| Item | Result |
|---|---|
| Reliable skew estimate | 92.5% of non-blank pages (≥3 of 4 estimators agree within 0.3°) |
| \|skew\| on reliable pages | p50 0.10°, p90 0.46°, p99 1.09°, max 3.04° |
| Pages rotated at 0.25° floor | 26% (416) |
| Blank pages | 247/1969 (12.5%), in only 13/291 PDFs; 31.9% of duplex backs, 0% of fronts |
| Blank detector | 0 false blanks, 0 missed; threshold 40 vs nearest non-blank 232 (5.8× headroom) |
| Spiral-coil pages | 34.5% (679) — left 352, right 327 |
| Hole-punch shadows | none found |
| Feeder dark strips | correlate with skew (r = 0.61) |

## A. Deskew

### A1. Survey (1722 non-blank pages; 1593 reliable)
\|angle\| percentiles: p50 0.10°, p75 0.26°, p90 0.46°, p95 0.64°, p99 1.09°, max 3.04° (#623, ruled notebook).
Share of reliable pages above: 0.15° → 42%; 0.25° → 26%; 0.3° → 21%; 0.5° → 8.7%; 1° → 1.4% (23); 2° → 3 pages.
Skew is predominantly negative (clockwise). Signed histogram (0.25° bins, bin start: count): −2.25:2, −1.75:2, −1.5:3, −1.25:8, −1:17, −0.75:49, −0.5:157, −0.25:358, 0:819, +0.25:121, +0.5:35, +0.75:14, +1:6, +1.25:1.

| side | mean | \|a\| p50 | p90 | p99 | >0.25° |
|---|---|---|---|---|---|
| front | +0.06° | 0.07 | 0.42 | 0.95 | 22% |
| back | −0.116° | 0.14 | 0.54 | 1.08 | 31% |

Front/back skew within a sheet are anti-correlated (r = −0.42); backs carry an extra ≈ −0.15° offset (p ≈ 1e-31 for mean, 3e-5 for \|a\|). No effect of native dpi (150/200/300 p90: 0.50/0.43/0.56).

### A2. Method comparison
Ground truth: 60 visually verified straight pages (seed 11; white, pink, yellow, blue, green paper; text, handwriting, graphs, diagrams) × 5 seeded rotations (3 in ±8°, 2 in ±2.5°) = 300 trials, white fill, 0.5 in L/R and 0.25 in T/B margin crop. Floor of ~0.1° from residual skew in "straight" pages. Self-consistency = error relative to the method's own estimate on the unrotated page.

| method | abs err mean | p95 | max | fail >0.3° | self-consist mean/p95/fail | ms/page (loaded) |
|---|---|---|---|---|---|---|
| projection | 0.128 | 0.335 | 1.43 | 8.7% | 0.071/0.186/1.7% | 510 |
| Hough | 0.166 | 0.397 | 1.53 | 14.0% | — | 2680 |
| LSD | 0.171 | 0.635 | 1.47 | 12.0% | 0.053/0.135/1.3% | 127 |
| blob minAreaRect | 0.140 | 0.322 | 1.24 | 6.3% | 0.155/0.375/10.7% | 48 |
| blob moments | 0.187 | 0.509 | 1.49 | 18.7% | 0.153/0.446/10.7% | 58 |
| gradient | 0.239 | 0.617 | 2.87 | 23.3% | 0.238/0.706/26.7% | 46 |
| **ensemble (final)** | **0.117** | **0.30** | **0.54** | **5.3%** | **0.080/0.18/0** | 363 CPU |

Ensemble: trials with \|a\| ≤ 5° and conf ≥ 0.6 (227/240 accepted, 94.6%). Weighted mean as in `geometry.py`: self-consistency 0.062 mean / 0.151 p95. End-to-end (estimate → correct → re-estimate, 104 trials): residual mean 0.031°, p95 0.053°, max 0.10°. All 5–8° trials were flagged unreliable/out of range, by design.

Bias found and removed: initially LSD looked best (0.079 mean), but rotated coloured pages on white fill create a crisp border at exactly the rotation angle; real scans have none. Margin cropping fixes it. Horizontal coil teeth also biased LSD toward 0°.

Real-page agreement (pairwise within 0.3°): projection 0.84, blob 0.83, LSD 0.80, Hough 0.80, gradient 0.76. In 184 projection-vs-LSD disagreements, the others sided with projection 128×, LSD 33×. LSD fails on handwriting/diagram pages (#76, #1937); projection on vertical text (#1870) → voting.

Paper-edge (wedge strip slope): available on 48% of pages (91% of \|a\| > 0.5°), MAE 0.29° vs consensus — diagnostic only. The scanner crops to the page; no background border is visible.

### A3. Rotation interpolation (32 trials, full-res 200/300 ppi, ±0.3–2.5°)

| method | SSIM | MAE | edge sharpness | ms |
|---|---|---|---|---|
| nearest | 0.9876 | 1.42 | 0.93 | 154 |
| linear | 0.9919 | 1.04 | 0.85 | 245 |
| **cubic** | **0.9953** | **0.69** | 0.94 | 389 |
| lanczos4 | 0.9959 | 0.64 | 0.93 | 1309 |
| 2× SS + area | 0.9928 | 0.95 | 0.88 | 1898 |

### Chosen
- `estimate_skew`: ≤100 ppi analysis, margin crop, 4 voters (projection 3, LSD 1.5, blob minAreaRect 1, gradient 0.5); largest cluster within 0.3° wins; confidence = (cluster size − 1)/3; search ±6°, accept ±5°. 0.36 s CPU/page.
- Rotate only if conf ≥ 0.6 and \|angle\| ≥ 0.25° (0.25° ≈ 0.9 mm displacement across 8.5 in; estimator noise 0.06–0.09°, p95 0.15–0.18°). Skip blank pages.
- `deskew`: `cv2.INTER_CUBIC`, same canvas, fill = paper colour (white after cleaning). 22/74/181 ms at 100/200/300 ppi.

## B. Blank-page detection (blank pages are dropped by default → false blanks are costly)

### Labels
326 pages labelled (union of 300 lowest core-ink and 300 lowest full-page ink), reviewed via boosted-contrast sheets and 0.75 in 1:1 crops: 247 BLANK, 79 NOT_BLANK. The remaining 1643 pages have >1.5% core ink (assumed NOT_BLANK). Non-obvious: NOT_BLANK #867 (purple "1A" in the corner), separator sheets with only a marker label (#894, #1044, #1356, #1581, #1220). BLANK: #891 (smudge, dot, dust streak), #1710 (bleed-through scribble), #905/#925 (dust rows), #871/#875/#879/#883 (dark top strip + dust). Page numbers sit as close as ~0.1 in to the bottom edge → no margin band can be excluded. No real single-faint-pencil-word page exists → sensitivity measured synthetically.

### Feature search (BLANK max vs nearest NOT_BLANK)
max-cluster mass 61.8 vs 69 (1.1×); total mass 80 vs 269 (3.4×); **total sharp-edge mass 25.3 vs 232 (9.2×)**. 150/200 ppi analysis did not help. Removing border-touching thin components deleted real grids → replaced by an explicit edge-artifact mask.

### Result
`blank_score` = sum of background-normalised ink over pixels with ink > 0.12 and Sobel gradient > 60, excluding edge artifacts, in 100-ppi pixel units. `is_blank` = score < 40. 116 ms CPU/page.

Confusion (all 1969): BLANK 247→247 blank, 0 missed; NOT_BLANK 1722→0 false blank. Nearest non-blank #867 = 232.2 (threshold is 0.17 of it, 5.8× headroom); highest blank 25.3 (1.6× below). Next non-blank: #1918 263, #1917 369, #1909 392, #1916 554, #894 564. 200 random 50/50 splits (threshold = geometric mean of train blank max and train non-blank min; mean 81, sd 13): 0 errors on every held-out half (validates threshold only; ink 0.12 / grad 60 chosen on these labels).

Synthetic sensitivity (24 real blank pages, 0.25 mm strokes, detection rate):

| mark | 40 gray levels darker than paper | 60 | 90 | 140 |
|---|---|---|---|---|
| word, 0.2 in tall | 0% | 100% | 100% | 100% |
| page number "14", 0.14 in | 0% | 0% | 58% | 100% |
| X mark | 0% | 0% | 25% | 100% |
| check mark / 0.4 in dash | 0% | 0% | 0% | ~edge |
| 0.06 in dot | never (same as dust, by design) | | | |

Limitation: thin faint strokes ≤ 0.3 in long and no darker than ~gray 170 on 250 paper fall in the dust range and would be dropped (originals are always backed up). A cluster-≥0.2 in variant (threshold 29, nearest non-blank 79, margin 2.7×) catches more but was rejected for lower margin.

## C. Borders and edge artifacts
- Spiral coil: 679 pages (34.5%), left 352 / right 327, never both; detected by edge dark-run periodicity (0.25 in pitch); band depth p5/p50/p95 = 0.27/0.35/0.41 in; ~4.8% of page painted on coil pages. Some pages show only coil tips (~0.1 in).
- Dark feeder strips (non-coil, non-blank): p99 strip depth >1 mm on 14–16% of pages (L/R), 4–5% (T/B); >2 mm: L 2.7%, R 2.2%, T 0.8%, B 0.5%; any side >2 mm 6.4%, >4 mm 1.2%. Strips are crooked-feed wedges (r = 0.61 with \|angle\|):

| \|angle\| | pages with strip >2 mm | median strip |
|---|---|---|
| 0–0.25° | 2.7% | |
| 0.25–0.5° | 6.4% | |
| 0.5–1° | 32.6% | 1.8 mm |
| 1–5° | 76.9% | 4.3 mm |

- No clipped content observed (unprinted corners appear as dark wedges). No hole-punch shadows (45 random non-coil pages inspected).
- Recommendation: coil + strip cleanup only. `clean_edges`/`edge_artifact_mask` paint the coil band and attached dark strips ≤ 0.35 in deep (+0.02 in growth) with paper colour; run before cleaning and deskew. Max painted fraction on non-coil pages 3.2% (coil-tip pages), median 0.43%. Caveat: a dark full-bleed graphic ≤ 0.35 in thick at an edge would be painted (none in corpus).
- Orchestrator spot check: page #153 (coil right, 10.3 Notes – Ellipses p3): coil removed, adjacent handwriting untouched; the top-edge strip removed was a 0.04 in edge line, not content.

## Constants (`geometry.py`)
`WORK_DPI=100, MAX_ANGLE=5, MIN_ANGLE=0.25, MIN_CONF=0.6, VOTE_TOL=0.3, BLANK_THRESHOLD=40`

## Limitations
- Straight-page ground truth only good to ~0.1° (absolute errors are upper bounds).
- 7.5% of non-blank pages get no reliable estimate and stay unrotated.
- Vertical-only text can mislead projection/blob; the vote usually rescues it.
- Blank labels beyond the 326 reviewed pages are assumed NOT_BLANK.
- Note: sample skew may underrepresent badly crooked small-batch Brother feeds the user reports; validate with live Brother scans.

## Panels (`panels/`)
`skew_top12_contact_1.png`, `skew_top12_contact_2.png`, `skew_0623_+3.04.png`, `skew_0133_-1.51.png`, `edge_cleanup_A.png`, `edge_cleanup_B.png`, plus `skew_<page>_<angle>.png`. Evidence in the study dir: `edge_s2.png`, `edge_s3.png`, `holes_s.png`, `crops_*.png`, `lab_*.png`, `dis_view.png`, `gt_c1.png`, `gt_c2.png`.
