# R6 — Fold creases (gray fold lines): detection and removal

> Figures referenced below are not published (they show private scans).

(Transcribed by the orchestrator from the R6 agent's final report; the agent could not write files at the end of its run. Prototype,
scripts and JSON results are in this directory. The motivating test document is a private, folded multi-page letter; its pages, crops
and OCR text stay out of the repo, only aggregate numbers are reported here.)

## Answer

Ship a conservative single-page fold remover: on the native RGB page **after edge cleaning, before `blank_score` and `clean_page`**,
remove only crease pixels inside a tracked ±0.017 in band, and only for a full-width, faint, isolated, non-periodic line at 1/3, 1/2 or
2/3 of the page height (tier A), plus two corroborated tiers (F: tri-fold family member, B: duplex partner).

- Test document: all 9 creases that survive the shipped cleaner are detected (p4, p6, p9×2, p11×2, p13×2, p14); 0 false detections on
  its 14 pages. The 7 faint creases the shipped cleaner already whitens are not claimed.
- Corpus: 1,946 of 1,951 evaluated pages have no detection. The 5 that fire are 3 genuine creases on pink paper (942, 943, 1042; tier B
  recovers a 4th, 1043) and 2 folded-over booklet-page edges (1901, 1902, removal touches blank area only).
- Removal brings crease survival near the noise floor on fold-only backs and to 5-10 % of the shipped cleaner's on the darkest crease;
  ink in the band loses ~0 on average.
- Cost: detection ≈ 0.12 s CPU/page (40-55 % of one `clean_page`), removal 0.02-0.06 s on fold pages only.

Premises the data corrected:
1. The thin blank-score margin of the fold-only backs p4/p6 (10.4/10.9 vs 15) is **not** the fold (fold band contributes 0.00/0.37); it is
   a staple/clip-like mark in the top corner. Fold removal does not change that margin.
2. The cleaner's line support is **not** what darkens the crease: with `line_in=0` the p13 upper crease survives at 8,414 px < 40 gl vs
   8,219 shipped (raw 3,439). The hysteresis gate seeded by the crease's shadow band does it.
3. The corpus does contain folded pages: pink half-folded "Formative 2" worksheets (942, 943, 1042, 1043).

Not safe / not handled: a perfectly straight, uniform, full-width gray hairline at a customary row is indistinguishable and is removed;
thin horizontal strokes within ±0.017 in of a detected crease can be erased; vertical folds are not handled.

## 1. Fold anatomy (test document, 300 ppi)

- Horizontal, tilt 0-0.75°, parallel to the paper edge, not to the print (p9 rules tilt 0.5°, its fold 0.0°).
- Rows: half fold 0.494-0.501; letter tri-fold 0.335-0.339 and 0.679-0.681. Front and back agree within 0.002-0.003 of page height.
- Cross-section: thin broken dark line 1-3 px, darkest ~150-160 gl (density ≤ 0.4); a mottled blue-gray shadow band ~0.2 in wide on one
  side, 2-5 gl mean contrast. Chroma/density ratio median 0.13 (pen ink 0.4-0.5).
- Straightness 0.4-7 px deviation; some creases dashed (~13 % gaps). Polarity is **not** inverted between the two sides.
- Look-alikes: printed table rules (6-10 px, uniform, stop inside margins, crossed by vertical rules); graph/notebook rules (periodic);
  booklet edges (2-3 parallel lines); highlighter (thick step); handwriting rows (intermittent). Only a crease spans edge to edge.

## 2. Negative set (corpus, structural proxies, montage-checked)

| class | pages | false detections | closest score (threshold 1) |
|---|---|---|---|
| isolated long rules | 313 | 0 | 0.53 |
| tables | 698 | 0 | 0.72 |
| graph paper | 175 | 0 (booklet edges excepted) | 0.84 |
| coloured handwriting | 648 | 0 | 0.67 |
| horizontal pen strokes | 430 | 0 | 0.58 |
| highlighter | 111 | 0 | 0.57 |
| pencil / faint | 185 | 0 | 0.72 |
| dashed | 22 | 0 | 0.35 |

No notebook-ruled or bubble-sheet pages exist in the corpus; covered synthetically (§8).

## 3. Detector (`fold.py`)

Stage 1 at 100 ppi: flat-field per-channel density, coloured pen ink excluded; ridge evidence for every (angle, row) in ±3° / 0.25° in
0.25 in windows along x, ink-masked ("observed" windows only). Score = min over criteria of value/threshold: presence ≥ 0.75 of observed
windows; ridge in both outermost windows (edge to edge); ink-free fraction ≥ 0.75; ≥ 0.4 windows observed; amplitude 0.02-0.14 and not a
step; continuity ≥ 0.55. Lines with another strong line within 0.1 page height are dropped (crowding). Pages < 5×7 in skipped.
Tiers: **A** score ≥ 1 within ±0.02 of 1/3, 1/2, 2/3; **F** score ≥ 0.6 at 1/3 page height from an A fold; **B** own ≥ 0.6 and duplex
partner ≥ 0.8 at the same row ±0.015. Native check: reject a dark (≥ 0.55), ink-isolated rule over > 0.25 of its length.

Ablation (corpus pages with a tier-A decision / test-doc recall of 9): ridge only 36 / 0; + edge-to-edge 83 / 7; + presence 18 / 9;
+ non-ink 14 / 9; + amplitude/step 7 / 9; + continuity 6 / 9; final 5 / 9. Leaving out edge-to-edge: 23 / 5 — it is the main
discriminator. Without the position prior 7 lines anywhere would fire.

Margin: highest non-fold score near a customary row 0.84 (graph page 852) vs 1.0. Synthetic recall (292 real crease strips pasted at
customary rows): 0.66 default (text 0.50, colour 0.59, faint 0.89), 0.78 with loose thresholds, 0 other detections — about a third of
faint creases over dense content are missed (they stay as the shipped cleaner leaves them, never worse).

## 4. Removal

Track the crease by dynamic programming within ±0.07 in of the line (ink masked); mask the ±0.017 in band except density ≥ 0.40 (+0.008 in
skirt), coloured ink, non-horizontal structure (structure tensor), and vertical strokes leaving the band; fill with local paper colour.

Synthetic (corpus pages × real crease strips; ink loss = stroke-core retention vs the uncreased reference; survival = crease footprint
px darker than 10/40 gl; shipped cleaner: 0.60-0.65 / 0.93-0.97):

| group (n) | ink loss mean / p95 / max | survival 10 / 40 gl |
|---|---|---|
| crossing ink (47) | −0.005 / 0.019 / 0.129 | 0.113 / 0.016 |
| customary row (44) | +0.010 / 0.103 / 0.570 | 0.130 / 0.032 |
| end-to-end, detector must fire (39) | +0.008 / 0.163 / 0.193 | 0.306 / 0.145 |

Losses ≥ 0.10 (3 of 130 instances) are thin horizontal strokes lying on the crease (a printed dashed rule, a thin table rule, pencil
fraction bars). Short ~12×2 px stubs remain where a crease crosses a vertical rule.

Real document, px darker than 10 / 40 gl in ±0.04 in, text-free columns:

| fold | raw | shipped | new (placement A) |
|---|---|---|---|
| p4 back | 7734 / 596 | 160 / 73 | 0 / 0 |
| p6 back | 12737 / 541 | 2717 / 1171 | 0 / 0 |
| p14 back | 10926 / 666 | 282 / 54 | 0 / 0 |
| p13 upper (darkest) | 10041 / 3439 | 13584 / 8219 | 1319 / 413 |
| p13 lower (through text) | 3107 / 646 | 585 / 276 | 238 / 125 |
| p9 ×2 | — | 2770/1250, 1533/662 | 947/408, 683/335 |
| p11 ×2 | — | 2593/1626, 2069/1074 | 220/137, 1258/653 |

Pages without a detection are pixel-identical to the shipped output; on fold pages, 0 px change beyond ±0.1 in of the fold (56 px of
≤ 1 gl on p9). On the 5 corpus pages that fire, changes extend ~0.2 in from the crease (pink shadow band no longer seeds the cleaner).
OCR F1 vs raw (tesseract, 200 ppi): p13 0.996 → 0.993 (1 word), p9 0.957 → 0.962, p11 0.947 → 0.898 (all differing words outside the
changed bands: layout analysis), fold strips 4 of 6 equal or better — small counts, no evidence of harm or gain.

## 5. Placement

Removal before cleaning (A) vs whitening after (C), 130 synthetic instances: survival 0.130/0.032 vs 0.184/0.034, ink loss 0.0096 vs
0.0056 (within noise); real p13 upper 413 vs 775 px < 40 gl. **A**: edge clean → fold detect → fold remove (in place) → blank score →
skew → clean. Disabling line support is not warranted.

## 6. Blank pages

Fold-only backs with removal: p4 10.37 (unchanged), p6 10.48 (10.85), p14 0.42 (0.79); p7, p8, p10, p12 ≤ 3.3. Corpus: no real-content page
newly drops; lowest real-content score stays 231.5 (page 867). Forced removal at all three rows on every page: minimum ratio 0.936.

## 7. Speed (one core, 8.3 Mpx, loaded machine)

`clean_page` 0.22-0.30 s; fold stage 1 0.12 s (0.096 from a 100 ppi RGB); native check + tracking 0.012-0.023 s and mask 0.016-0.034 s
per fold page. Stage 1 is dominated by numpy sub-window sums.

## 8. Hypotheses tested

**OCR (tesseract word boxes).** As protection it does not reduce ink loss (0.0094 vs 0.0096; the worst cases have no confident words) and
worsens crease survival (< 40 gl 0.052-0.068 vs 0.032), at 0.8-1.1 s CPU per page. On this user's content, confident boxes cover 0.46 of
printed-text ink, 0.22 of coloured handwriting, 0.04 of pencil. As evidence that a line is not text, box coverage of crease and non-crease
candidate lines overlaps (median 0.01). Not used; "no OCR text" must not drive dropping or removal.

**Duplex corroboration.** The crease is at the same row on both sides (6-9 px), same polarity. Geometric (black-print) pairing via
bleed-through NCC works only with strong bleed (true sheets z 5.9-9.5, weak ones 0.6-2.7, non-sheets ≤ 3.7) — use PDF order / the
existing `duplex.py` pairing. As a gate (own ≥ 0.6, partner ≥ 0.8) it adds exactly one hit corpus-wide, the real fold on 1043.

**Adversarial (synthetic on real blank pages).** Notebook ruled paper, 3 in pencil line, highlighter band, underlines, full-width dashed line
(best 0.92-0.93): not detected. A full-width uniform gray hairline at 0.5 is detected in most variants.

## 9. Integration API

```python
cands, (w_in, h_in) = fold.line_candidates(img, dpi, P)        # stage 1, reusable as a duplex partner's evidence
folds = fold.detect_folds(img, dpi, P, partner=partner_cands)
mask  = fold.fold_mask(rgb, dpi, folds, P)
rgb2  = fold.remove_folds(rgb, dpi, folds, P, mask=mask)
```
All thresholds are named in `FoldParams`.

## 10. Limitations

Little positive data (9 creases from one scanner, 4 pink corpus pages); thresholds tuned on the same data. Recall ~2/3 on faint creases
over dense content. Customary-row prior (quarter folds only via tier B). Straight uniform hairlines at customary rows are removed.
Thin horizontal strokes on a crease can be erased. Pink-paper creases are only partly cleaned (a light pinkish band remains). Residue
where creases run through text rows. Booklet edges treated as folds (harmless, cosmetic dashes remain). No vertical folds.

## Orchestrator review

Inspected the private panels (p13 upper: crease fully removed where the shipped cleaner had turned it into a solid dark line; p13 lower:
the text line next to the crease intact, crease mostly gone; p11 fold through a text row: much lighter, faint residue under one word) and the committed
`synth_text.png` (crease through printed text: lighter than shipped, residue through the glyph row, stubs at vertical rules) and
`corpus_detections.png`. Committed JSON/scripts checked for private-document text: none.

## Panels (`panels/`)

`neg_{graph,table,rule,dashed,highlighter,pencil,handwriting,hstroke}.png`, `adversarial.png`, `synth_{text,colour,faint}.png`,
`corpus_detections.png`.
