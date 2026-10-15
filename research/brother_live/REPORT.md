# scanpp on a live Brother ADS-2200 duplex scan (22 pages, 11 sheets)

> Figures referenced below are not published (they show private scans).

Evaluator run, no source changes. Tested **git HEAD a01a781d3631** plus the uncommitted working-tree changes present at snapshot time
(clean.py, geometry.py, pipeline.py modified; diff sha 06d3a7611dfe). The tests ran from a frozen copy of `src/` taken at that moment
(a scratch snapshot of `src/`), so concurrent edits did not affect them. A re-run of the same input gave identical output (page 5 render md5 equal).
Scratch work dir (not published) (inputs `raw/`, `build/*.pdf`, outputs `out/{A,B,A_ocr,A_keep}.pdf|json`, analysis scripts `an/`).

Inputs: (A) `brother_q75.pdf` = Pillow q75 4:2:0 JPEG per page at 300 dpi (13.56 MB); (B) `brother_lossless.pdf` = Flate RGB (148.07 MB).
Scan size is 2549 px wide, 3254-3299 px tall (the Brother crops to the detected sheet length), so the letter-area setting is not clipping lengthwise.
Page map: out pages 1-16 = raw 1-16, out 17 = raw 19, out 18 = raw 21. Dropped: raw 17, 18, 20, 22.

## 1. Summary of verdicts

| feature | verdict | key numbers |
|---|---|---|
| Bleed-through, paper tint / pencil / printed text | **pass** | colored-paper pages: raw bleed region 2.1-2.8 gray levels darker (p95 8-11), output 0.00 (p95 0); white pages ~0-1.5 gl raw, 0 out |
| Bleed-through, saturated highlighter / marker (bleed test sheet back, p2) | **FAIL** | blue and purple marker bleed from p1 is kept AND amplified: mean darkness on the marks 57->89 and 40->77 gray levels, area darker than 40 gl 1836->2899 and 2487->5282 |
| User handwriting on p2 incl. "very light" notes | **pass** | ink recall 0.999-1.000 for all notes, output darker than raw (mean d 98 -> 110) |
| Skew | **pass** (one concern) | max |angle| 3.08 deg (p21), 2.35 (p19); all applied (conf 0.67); residual <= 0.04 deg. p3 (0.6-0.7 deg) left uncorrected in (A) |
| Skew > 5 deg handling | **not exercised** | no page in this stack exceeds 3.08 deg; a rescan with a >5 deg sheet is needed |
| Blank-page drop | **pass** | 4 dropped (scores 0.14, 1.5, 5.0, 9.6) all truly blank; no non-blank page dropped; see section 5 for the planted pencil line |
| Colored paper | **pass** (cosmetic concern) | yellow and blue to RGB 255/255/255; pencil Lab chroma median 0-3.6 (raw 18-34), p95 <= 11; purple pen X and p1 colors kept; top-edge residue on 6 pages |
| Small print | **pass** | T&C page: 3.0% (A) / 2.2% (B) of 1709 ref words lost vs 2.2% lost by the q75 input alone; legible at 110 ppi |
| Highlighter | **pass** | green kept at 150 ppi (e1); text over it crisper than raw; p2 yellow/orange notes kept |
| Edge / boundary cleanup | **pass** (concern) | side strips and backing cleaned; crooked-sheet corners clean; no text painted; top-edge blotches on yellow/blue pages |
| (A) vs (B) | q75 matters slightly | bytes 1.486 vs 1.539 MB; mean SSIM 0.982 (min 0.848, p3); p3 deskewed only in B; 3 pages gray in A, color-coded in B |
| Timing | ok | 15-20 s wall for 22 pages at --jobs 16 on a machine at load ~15 |

Issues ranked by severity are in section 9.

## 2. Per-page table (run A)

| raw p | out p | dropped | blank_score | skew est (deg) | conf | applied | residual (out, 4-vote) | scheme | edge painted | bytes | lost words A / B / ref |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | no | 58247 | +0.52 | 1.00 | yes | +0.04 | e1 | 1.5% | 260133 | 20 / 12 / 425 |
| 2 | 2 | no | 2708 | +0.61 | 0.33 | NO | +0.13 | e3x | 1.5% | 32711 | 0 / 0 / 2 |
| 3 | 3 | no | 11706 | +0.32 | 0.33 | NO | +0.30 | e3x | 1.1% | 78741 | 3 / 5 / 38 |
| 4 | 4 | no | 12465 | -0.59 | 0.67 | yes | +0.03 | e3x | 1.3% | 92932 | 3 / 1 / 49 |
| 5 | 5 | no | 5617 | +0.52 | 1.00 | yes | -0.01 | e3x | 1.2% | 52465 | 0 / 0 / 29 |
| 6 | 6 | no | 7639 | -0.85 | 1.00 | yes | +0.04 | e3x | 1.4% | 57666 | 1 / 1 / 36 |
| 7 | 7 | no | 13774 | +0.46 | 1.00 | yes | +0.02 | e3x | 1.6% | 66164 | 0 / 3 / 101 |
| 8 | 8 | no | 12781 | -0.29 | 1.00 | yes | -0.01 | e3x | 1.7% | 75314 | 0 / 1 / 98 |
| 9 | 9 | no | 9455 | +0.42 | 1.00 | yes | +0.02 | e3x | 1.7% | 57389 | 2 / 3 / 65 |
| 10 | 10 | no | 2670 | -0.28 | 0.67 | yes | +0.01 | e3 | 1.7% | 24743 | 0 / 1 / 18 |
| 11 | 11 | no | 24172 | +0.29 | 1.00 | yes | +0.00 | e3x | 1.5% | 69525 | 0 / 0 / 117 |
| 12 | 12 | no | 13011 | -0.41 | 0.67 | yes | +0.02 | e3x | 1.7% | 45001 | 0 / 0 / 45 |
| 13 | 13 | no | 19914 | +0.36 | 1.00 | yes | +0.01 | e3x | 1.3% | 84128 | 5 / 5 / 99 |
| 14 | 14 | no | 22194 | -0.32 | 1.00 | yes | +0.01 | e3x | 1.5% | 106604 | 2 / 1 / 99 |
| 15 | 15 | no | 16929 | +0.28 | 1.00 | yes | -0.00 | e3x | 1.3% | 87638 | 0 / 2 / 80 |
| 16 | 16 | no | 4962 | -0.37 | 0.67 | yes | +0.02 | e3x | 1.5% | 44087 | 1 / 1 / 17 |
| 17 | - | DROPPED | 0.14 | - | - | - | - | - | 1.0% | 0 | - |
| 18 | - | DROPPED | 1.51 | - | - | - | - | - | 1.0% | 0 | - |
| 19 | 17 | no | 32478 | -2.35 | 0.67 | yes | +0.04 | e3 | 2.9% | 199176 | 52 / 38 / 1709 |
| 20 | - | DROPPED | 4.98 | - | - | - | - | - | 2.9% | 0 | - |
| 21 | 18 | no | 4504 | -3.08 | 0.67 | yes | +0.04 | e3 | 3.3% | 37751 | 2 / 2 / 130 |
| 22 | - | DROPPED | 9.59 | - | - | - | - | - | 3.1% | 0 | - |

"residual" = the pipeline's own 4-estimator vote re-run on the 300 ppi output render. For non-applied pages (p2, p3) conf is only 0.33, so that
number is not trustworthy; an independent projection-profile search gives p3 about 0.58 deg and p2 about 1.2 deg of unreliable texture slant.
Corrected pages: independent projection profile on the output gives 0.00 deg for every one of them.

## 3. Runs

| run | wall | CPU | in -> out | pages | notes |
|---|---|---|---|---|---|
| A (q75, defaults) | 20.3 s (re-runs 15.0 s, 17.9 s) | 106 / 78 / 84 s | 13.56 MB -> 1.486 MB (9.1x) | 22 -> 18 | |
| B (lossless) | 30.8 s | 96.6 s | 148.07 MB -> 1.539 MB | 22 -> 18 | wall is dominated by Flate decode and I/O |
| A --ocr | 41.6 s | 125.9 s (ocr 51.4 s) | -> 1.536 MB (+3.4%) | 22 -> 18 | render pixel-identical to A (max diff 0); 3819 words extracted |
| A --paper keep | 13.8 s | 68.8 s | -> 1.837 MB (+24%) | 22 -> 18 | paper RGB median kept (yellow 255/248/176, blue 180/248/251), std ~2 |

Load average was 13-15 on 16 cores during all runs (shared machine), so walls are noisy. Per-stage CPU, run A (106.1 s total): clean 57.3 s (54%),
encode 26.4 (25%), edge 8.2 (8%), deskew 5.0 (5%), skew 4.9 (5%), blank 2.3 (2%), decode 2.0 (2%). The slowest single page is 7.2 s CPU (p14), so about
7 s is the floor at 16+ idle cores. OCR is about 2.9 s CPU per page. Schemes: p1 e1 (highlighter route), p10/p19/p21 e3 (gray), the rest e3x. Edge painted fraction
1.1-1.7% on ordinary sheets, 2.9-3.3% on the crooked sheets p19-p22.

## 4. Bleed-through

Method: each page was straightened with its applied angle, the mirrored back was registered to the front (template matching on faint-layer vs back-ink
at 100 ppi, search window +-6 px, shifts came out within 0-5 px), region = back ink dilated 0.03 in minus front ink dilated 0.06 in, measured as gray levels darker
than local paper in raw vs output (`an/bleed.py`, log `an/bleed_log.txt`). Registration z-scores are weak on colored paper (1-4), so those numbers are
region-based approximations. Region measure also contains real faint content, so the residual counts are upper bounds.

* White pages (p1/2, 7-12): raw bleed-region mean 0.0-1.5 gl (p95 <= 6); output mean 0.00, p95 0. Pencil bleed on the white Unit 3 pages is not visible in the output.
* Yellow/blue pages (p3-6, 13-16): raw region mean 1.8-2.8 gl, p95 8-11, 13k-79k px darker than 10 gl; output mean 0.00-0.08, p95 0. The remaining
  few hundred to ~1400 px darker than 20 gl were inspected (`bleed_residuals_colored_paper.png`): they are real faint pencil (p4), crease/crumple shadows (p13, p16), not bleed.
* **Bleed test sheet (p1 front macOS page with highlighter + pens, p2 back with notes).** p1's blue and purple marker underlines bleed through
  onto p2 as speckled streaks at the mirrored positions. In the output they remain and become denser blobs (`bleed_p2_crops_1to1.png`):
  blue streak mean darkness 57 -> 89, purple bars 40 -> 77 gray levels. This is the worst finding of the stack. The cleaner treats them as ink because the marker is saturated and dark enough
  (raw peak d 150-205). Orange/yellow/green highlighter from p1 did not produce visible bleed on p2, and p2's handwriting (red, green, pencil) does not bleed onto p1 (p1 output is clean).
  The faint arrow/bracket at the top-left of p2 (bleed of p1's blue "Is [fn] a key..." annotation) also remains in the output at lower visibility.
* **Notes on the back survive.** All six notes on p2 kept: ink recall (raw d>25 px with an output ink px within 2 px) 0.999-1.000; mean darkness raw -> out:
  header 138 -> 155, "Will orange bleed through more?" 121 -> 142, "Green is fairly light" 99 -> 110, "This is also fairly light" 98 -> 110, "0.3 mm pencil" 73 -> 102. Yellow note
  survives (visible in `highlighter_p1_p2.png`). Nothing is lost and nothing is weakened.

Panels: `bleedtest_sheet1_overview.png`, `bleedtest_sheet1_boosted.png` (contrast x3, shows bleed in raw/out), `bleed_p2_crops_1to1.png`, `bleed_residuals_colored_paper.png`.

## 5. Blank pages and the planted pencil line

Dropped: p17 (score 0.14), p18 (1.51), p20 (4.98), p22 (9.59); threshold 40. Verified with flat-field boosted views (paper 255, d=0.05 -> 128; `dropped_pages_proof.png`):
p17 and p18 only dust and edge shadows; p20/p22 are the backs of the T&C sheets and carry a horizontal fold/crease shadow at y ~ 1600 px plus dust,
no marks. No non-blank page was dropped (the lowest kept page is p10 at 2670, p21 at 4504, p2 at 2708).
Blank score of truly blank pages: 0.14-9.6 here; across the 247 labeled corpus blanks: median 0.0, 99th pct 3.6, max 4.85, none >= 10.

**The planted line.** Best candidate: **raw p18** (back of sheet 9), a short graphite stroke, 8x20 px bounding box at x=270-282, y=217-242 px (0.9 in from the left, 0.75 in from the top),
length 0.07-0.09 in, width ~4 px (0.013 in, slightly curved), darkness peak 135 gray levels below paper (paper = 255, stroke center ~120), mean 77, neutral gray.
Its blank_score is **1.51 (page dropped; threshold 40)**; the stroke is diluted at the 100 ppi analysis resolution (peak ink 0.24, 10 px).
A second line-like object is on **p17**: a perfectly straight 1 px wide vertical streak at x = 994, y = 0..492 px (1.64 in), 22-35 levels below paper (peak 33, mean ~30). It starts at the top edge, does not wobble and exists on no other page, so it looks like
a scanner or paper-fiber artifact rather than pencil, but if that is the planted line the page-17 score is 0.14 (it vanishes at 100 ppi, below the ink threshold 0.12). Please confirm which is planted.
Either way the page was dropped, which the user accepts. Panel: `blank_line_p18_features.png`.

**What separates the line from dust (components of ink map > 0.12 at 150 ppi, 247 corpus blanks = 183 components, plus this stack).**
Features of the p18 stroke: npx 22, length 0.068 in, width 2.2 px, elongation 4.7, mean ink 0.183, peak 0.240, max gradient 110 (8-bit gray levels per px).
* Corpus blank components: length median 0.03 in (max 0.36), elongation median 4 (max 54), mean ink median 0.13 (max 0.31), peak median 0.14 (max 0.56).
  Length and elongation do NOT separate (long thin dust/fibers up to 0.36 in and elongation 54 exist). Darkness alone does not separate (24 compact blobs are darker).
* Compact dark specks are round (elongation < 3) and very short. **Elongated components (elong >= 3, npx >= 15; n = 31) have mean ink <= 0.154 and peak <= 0.199; the stroke has 0.183 and 0.240: 19% and 20% above the max.**
  No labeled blank page has a component with (elong >= 3 AND npx >= 15 AND mean >= 0.16) or (... AND peak >= 0.22) or (... AND max gradient >= 100): 0 of 247 pages flagged; the stroke passes all three.
  Single-feature rules without elongation flag 5-21 of 247 blank pages (e.g. mean >= 0.17: 21 pages).
* Margin is thin (about 20%) and rests on ONE positive example and 31 elongated corpus components, so it is a design lead, not a validated rule. Sharpness alone (max gradient)
  is not discriminative (stroke 110 sits inside the blobs' range 5-237).
* At native 300 ppi the stroke is distinct (4 px wide, peak 135 gl, vs dust of 1-2 px and peaks mostly < 60), but the corpus cache is only 150-200 ppi, so that was not tested on corpus.
* This stack's own dust (p17-p22): largest components <= 33 px at 150 ppi with mean ink <= 0.18; the dark compact blobs at the very top edge of p20/p22 (mean 0.28-0.31, elong < 2) are edge shadow, not line-like.
* Why a score fix is hard: the p18 stroke accounts for ~1.5 of 40 points; to cross the threshold on mass alone the mark would need ~25x more ink. A separate "has elongated dark component" veto would be needed.

**Calibration of `blank_score` in user terms** (synthetic marks painted onto real blank pages p17/p20/p22 with the measured pencil profile; real handwriting words cropped from p2; base pages
score 0/5/9, results given for p17 base and range over the three bases; `an/calib.py`, `an/calib.json`):

| mark | blank_score | dropped at 40? |
|---|---|---|
| (a) planted-size line 0.07 in, dark (135 gl, 4 px) | 5-15 (real planted stroke on p18: 1.5) | yes |
| 0.25 in dark line | 19-29 | yes |
| 0.5 in dark line | 37-47 | borderline |
| 1 in dark line | 74-85 | no |
| (b) 2 in dark pencil line | 148-162 | no |
| (b) 2 in light pencil line (60 gl) | 65-75 | no |
| 4 in dark / light | 294-313 / 131-140 | no |
| 1 in light pencil line (60 gl) | 33-42 | borderline |
| (c) one handwritten word "pencil." (real, 0.3 mm pencil) | 40-50 | borderline, dropped on clean base |
| (c) one handwritten word "light." | 62-71 | no |
| (c) phrase "0.3 mm pencil" | 102-111 | no |
| (d) page number "2" (typeset ~0.08 in, black / gray 60) | 20-36 / 20-30 | **yes, dropped** |
| (d) "12" | 51-60 / 37-47 | borderline |
| (d) "Page 2 of 2" footer | 138-148 / 103-113 | no |

Rule of thumb: the score is ~74 per inch of dark pencil (peak 135) line, ~33 per inch of light pencil, ~35-50 for one short word. 40 therefore drops a single pencil word or a lone
one-digit page number, and a ~1.2 in light pencil line. The scores of all non-blank content in the corpus labels have a 5th percentile of 96 (5 of the 79 labeled non-blank pages are <= 100 and 2 are < 40, i.e. already dropped
by the current threshold), so a `--blank-threshold` of 10-15 would still clear every labeled blank (max 4.85 corpus, 9.6 here) while keeping every mark above roughly 0.2 in of dark line / a single digit (synthetic digits are thin; real 10 pt digits score higher).
Synthetic marks are idealized (perfect AA, no noise) and typeset digits are 2 px strokes; treat the numbers as +-30%. A rescan with a real pencil line series (0.5, 1, 2 in; dark and light) and a lone page number would anchor this.

## 6. Skew

* Max measured angle **3.08 deg (p21), 2.35 (p19)**; all others 0.28-0.85 deg. The user's crooked sheets are p19-p22 (p20/p22 dropped as blank, p21 -3.08, p19 -2.35). No page reached 5 deg.
* The 0.6 confidence rule: p19 and p21 were 0.67 (3 of 4 estimators agree) and were corrected; the residual after correction is 0.04 deg by the same estimator and 0.00 deg by an independent projection-profile search. Before/after in `skew_before_after.png`.
* **Uncorrected pages:** p2 (est +0.61, conf 0.33; handwritten sparse page; independent projection disagrees) and **p3 (est +0.32 conf 0.33 in A; an independent estimate and the estimator on the unpainted raw give 0.58-0.69 deg).** In run (B) p3 was deskewed (0.692). So one real, small 0.6 deg tilt on p3 stays in the q75 output: minor (about 0.7 mm over a 6 in line) but it is a case where JPEG q75 changed a vote.
* The >5 deg rule (reported unreliable and left alone) was not triggered; this stack contains nothing to test it. If a heavier-crooked sheet (6-10 deg) is available, a rescan would test it.
* Corner clipping by the scanner (letter area): crooked sheets p19/p21 have the paper edge crossing the scan frame. Edge-depth profiles (`crooked_corners_raw.png`): on p21 the top paper edge is above the frame across the left ~half of the page
  (depth zero for x < ~1300 px, slope 0.06) and the bottom-right is similarly cut, consistent with up to ~0.25 in of top-left and ~0.1 in of bottom-right paper margin outside the scan area. Dark ink bounding boxes in raw:
  p19 left 0.42 in / right 0.54 / bottom 0.44; p21 left 0.73 / right 0.60 / bottom 0.44; no text was clipped. The 0.00 in top clearance on p19/p21 is the staple/hole smudge at the very top of the sheet, which sits in the clipped corner. Mildly crooked sheets (p6, p13, p14, p15, p16) also touch the frame by 0-20 px at one top corner without losing content.

## 7. Colored paper, small print, highlighter, edges

* Colored paper: output paper RGB 255/255/255 for all yellow (raw median 255/243-249/172-177) and blue (177-183/241-250/253-255) pages. Pencil strokes (pixels with Y < 150) on yellow: Lab chroma median 3.2-3.6, p95 7-11 (raw 18-34 incl. paper tint);
  blue: median 0.0-2.2, p95 4-6 (raw 19-20). No color cast. Purple pen X on p13 and the colored p1/p2 pens remain colored (colored pixel count 1353 vs raw 1430 on p13). Panels `paper_yellow_blue_pencil_1to1.png`, `paper_colored_pens_1to1.png`.
  The scanner clips the paper's dominant channel at 255 (yellow R=255, blue B=255), which the cleaner handles.
* **Small print:** OCR word recall vs the lossless 300 dpi raw (tesseract, conf >= 80 ref words, rotation-compensated positions, `an/lw.py`): T&C p19 1709 ref words: lost A 52 (3.0%), B 38 (2.2%), raw-q75 render alone 38 (2.2%).
  p21 130 words: A 2, B 2. macOS shortcuts p1 425 words: A 20 (4.7%), B 12 (2.8%), q75 raw alone 7 (1.6%) - p1 is the weakest because the highlighter route encodes it as a 150 ppi q55 JPEG (e1). All pages: lost A 91, B 76, q75-only 64 of 3157 reference words (2.9% / 2.4% / 2.0%).
  Lost words are mostly 5-6 pt UPS fine print and keyboard-shortcut labels. Visual crops at 110 ppi 100%, 110 ppi 200% and 160 ppi (`smallprint_TC_p19.png`, `smallprint_macos_p1_highlight.png`): the output is crisper than raw, fine print readable at 110 ppi 200% and at 160 ppi.
* **Highlighter:** p1 green/blue/purple/red and p2 orange/yellow/green notes all survive; text over green highlighter is darker and cleaner than raw (`highlighter_p1_p2.png`, `smallprint_macos_p1_highlight.png`).
* **Edges:** crooked sheets p19/p21 and short sheets: output border bands (40 px) are 0.0-0.3% non-white except the top band of yellow/blue pages (p3 5.5%, p5 3.5%, p6 3.7%, p13 6%, p14 8.6%, p15 6%, p16 5%) - lavender/pink fragments and gray-blue blotches from the crumpled top
  corner and shadow (`edges_top_residue.png`). Cosmetic, outside the text area; the backing strips and black side bars are gone and no printed content was painted over (raw ink bounding boxes are 0.23-0.73 in clear of every edge).
* **(A) vs (B):** output bytes 1,485,763 vs 1,538,701 (A is 3.4% smaller); mean SSIM of 220 ppi renders 0.982 (17 of 18 pages >= 0.965; p3 0.848 because B deskewed it and A did not; p17 0.98 from the gray/color scheme); lost words 91 vs 76 of 3157; pages 10, 19, 21 were e3 gray in A but e3x color in B
  (the lossless scan has scanner chromatic fringes on the black text, chroma > 35 on < 4 dark px; q75 4:2:0 smoothing makes the pages correctly gray); blank decisions identical. Conclusion: the scanner's q75 JPEG does not matter materially except through the skew vote on p3 (and small OCR loss); simple-scan q75 input is adequate.

## 8. Brightness / contrast (simple-scan 17 / 63)

* **Raw paper is already clipped to 255 by the scanner with no software adjustment.** White-paper pages: median RGB 255/255/255 and 76-99.8% of pixels (excluding content) have all three channels exactly 255; yellow paper R=255 (G ~243-249, B ~172-177), blue paper B=255 (R 177-183, G ~241-250).
  This matches the historical corpus assumption (paper pre-clipped to 255).
* Faintest real content: the p2 "very light" notes: darkest pixel 186-205 levels below paper (min channel), mean stroke ~78-98 gl; pencil "0.3 mm" peak 171, mean 57-73. Pale yellow highlighter ink is the faintest at only 23-47 gl in the min channel. No content is near the paper level.
* `scanimage -d 'BrotherADS2200:libusb:001:026' -A` lists only -l, -t, -x, -y, --resolution, --mode, --source. **It exposes no brightness, contrast or gamma option.**
* simple-scan `src/scanner.vala` (master, lines 1219-1250): it looks up `Sane.NAME_BRIGHTNESS` / `NAME_CONTRAST` and sets them only `if (option != null)`, scaling -100..100 to the device range; there is no software brightness/contrast anywhere in scanner.vala, page.vala, book.vala (grep clean).
  So on this Brother the 17/63 settings do nothing; the 22 raw pages are the scanner's own default rendering. (The user's Epson GT-8300 does expose `--brightness -4..3` and no contrast option; simple-scan 17 maps to about 0 there, contrast 63 has no effect.)
* **Conclusion:** no need to change them for the Brother, since they are not applied; resetting to 0/0 is harmless and avoids a surprise if a different scanner is used. Nothing would improve by changing them: the paper is already at 255 and the faintest content is >= 23 gl.
  A rescan is not needed for this.

## 9. Issues, ranked by severity

1. **Highlighter/marker bleed-through is kept and amplified** (p2 back of the bleed-test sheet, blue and purple marker from p1; mean darkness 57->89 and 40->77, area >40 gl 1.6-2.1x). Bleed of saturated marker needs a bleed test against the mirrored page (or saturation-and-speckle heuristics) before cleaning.
2. **p3 is not deskewed in the realistic (A) path** (0.6-0.7 deg left; B corrects it); the vote on p3 flips between q75 and lossless (conf 0.33 vs 0.67). Same class: p2 conf 0.33 not corrected. Minor.
3. Short, light marks drop pages: one handwritten pencil word scores 40-50, a lone one-digit page number 20-36, a 1 in light pencil line 33-42 (threshold 40). The user accepts line drops; page numbers and single words are a more likely loss. Configurable `--blank-threshold` recommended; 10-15 is safe on all 247 labeled blanks and this stack.
4. p1 (macOS page) is encoded as a 150 ppi q55 JPEG (highlighter route); it has the highest lost-word rate (4.7% vs 1.6% for the q75 input alone). Readable but less sharp than the e3 pages.
5. Top-edge residue on yellow/blue pages (lavender/pink fragments, gray-blue crumple blotches; p3, p5, p6, p13-p16) and crease shadows on blue p13/p16: cosmetic.
6. Scanner clips up to ~0.25 in of paper margin at the top-left and ~0.1 in at the bottom-right of the crooked sheets p19/p21 (no text lost; the staple/hole mark at the top of p19/p21 is in the clipped corner).
7. For the lossless path (B): 3 white pages (p10, p19, p21) are color-coded by scanner fringes, +20-30% bytes on those pages; q75 hides this.

## 10. Panels (`research/brother_live/panels/`)

* `bleedtest_sheet1_overview.png` - p1 and p2 raw vs out (1/5 scale); `bleedtest_sheet1_boosted.png` - same with contrast x3 (bleed visible)
* `bleed_p2_crops_1to1.png` - 1:1 crops of p2 raw / out / mirrored p1: blue and purple marker bleed kept and darkened
* `bleed_residuals_colored_paper.png` - residual regions on p16, p4, p13 with the mirrored back ink: crease shadows, real pencil
* `skew_before_after.png` - p21 (-3.08 deg) and p19 (-2.35 deg) raw vs out; `crooked_corners_raw.png` - corner crops (contrast boosted) showing the sheet edge crossing the scan frame
* `dropped_pages_proof.png` - flat-field boosted raw p17, 18, 20, 22; the planted stroke (p18, circle) and the p17 vertical streak (box)
* `blank_line_p18_features.png` - the planted stroke raw 1:1 x8 / contrast, and the feature distributions vs 247 blank pages
* `paper_yellow_blue_pencil_1to1.png`, `paper_colored_pens_1to1.png` - 1:1 crops, yellow p4 / blue p14 pencil, colored pen
* `smallprint_TC_p19.png`, `smallprint_macos_p1_highlight.png` - 110 ppi 100%, 110 ppi 200%, 160 ppi, raw vs out
* `highlighter_p1_p2.png` - highlighter and notes, raw vs out
* `edges_top_residue.png` - top 110 px of p3, p14, p16 raw vs out

## 11. Where a rescan would help

(1) a sheet fed at 6-10 deg to exercise the >5 deg "unreliable, left alone" path; (2) a pencil line series (0.5/1/2 in, dark and light) and a lone page number on blank backs to anchor the blank calibration, plus
confirming whether p17's straight streak or p18's stroke was the planted line; (3) the same bleed sheet with only highlighter on the front (one color per band) to get a per-color bleed curve.
