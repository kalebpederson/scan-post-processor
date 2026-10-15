# R5 — Duplex-aware bleed-through suppression

> Figures referenced below are not published (they show private scans).

(Transcribed by the orchestrator from the R5 agent's final report; the agent could not write files at the end of its run. Prototype and
scripts are in this directory. `src/` and `tests/` were not touched; scanpp was imported read-only from a frozen copy.)

**Scope.** Only marker-grade sources (chromatic ink wider than a pen stroke) are suppressed. Thin-pen, black-print and graphite ghosts
stay with the single-page cleaner. All thresholds were tuned on the same corpus they are evaluated on. Live evidence is thin: 2 of the
11 Brother sheets are accepted, and sheet 1 is the only strong marker case.

## 1. Pair detection

R2's method (A's faint layer correlated with B's mirrored ink) does not work as a detector: z >= 10 on 7 % of non-pairs, up to z 130
(sparse and layout-similar pages).

What works (`duplex.register_pair`, `accept`, `pair_sheets`):
- Bleed comes from wet colour ink, so match **chroma maps**; black print and graphite barely bleed.
- The NCC peak must be **unique** (peak minus best peak outside a 9x9 px neighbourhood = "gap").
- The two directional registrations must **agree**.
- The source page must carry **colour ink**.

Registration: 50 ppi `matchTemplate` over ±1 in in both directions; then ECC affine (scale within 1.2 %), falling back to rigid ECC,
then translation. Output `M_ab`, a 2x3 native-pixel matrix including the mirror.

Gate (`GATE`): accept if
- two-way: both gaps >= 0.05, registrations agree to <= 0.12 in, ncc >= 0.20; or
- one-way: gap >= 0.06 and ncc >= 0.50;
- and in both cases the source has >= 400 px (50 ppi) with chroma > 0.15.

A page in two accepted pairs keeps the one that beats the rival by 1.5x on ncc·gap, else both are dropped. Parity is not reliable
(11 of R2's 19 sheets are (2,3)-type; 39 of 138 accepted adjacent pairs are (even, odd)), so every consecutive pair is tested.

| set (1,969 pages, 291 PDFs) | n | accepted |
|---|---|---|
| adjacent pairs | 1,659 | 138 (8.3 %), in 95 docs; 0 page conflicts |
| far pairs (same doc, 2+ apart) | 5,541 | 1 (a real sheet) |
| cross-document pairs | 38,144 | 1 (a real sheet scanned into two PDFs) |
| **false accepts** | 43,685 | **0** (95 % upper bound 7e-5) |

True pairs: R2's 19 verified sheets → 12 accepted (the 7 rejected are the weakest ghosts, which the cleaner already removes). Brother
stack: sheets 1-2 and 5-6 accepted, the 10 adjacent non-pairs rejected; sheets 3-4, 11-12, 15-16 register (agreement 0.03-0.05 in) but
ncc 0.18-0.28 < 0.50; the rest have no colour ink. Sheet 1 after a q75 4:2:0 re-encode is still accepted (gap 0.08, ncc 0.56).

Margins: nearest negative ncc 0.39 vs 0.50; without the ncc test 4 negatives would pass. The colour-source test is the largest safety
net (a near-blank bubble-sheet pair had gap 0.39, ncc 0.48, but 1 colour px). Sheet 1's own margin is thin (gap 0.086 vs 0.06, ncc 0.60
vs 0.50); a miss means "do nothing", i.e. today's behaviour.

Flips: on 1,677 adjacent pairs the horizontal mirror gives 68 pairs with gap >= 0.15; vertical flip, 180° and no flip give 1 each
(background). Nothing needs a flip. Blank backs: no colour source → no pair (Brother blank backs gap <= 0.01).

## 2. Suppression rule (`duplex.bleed_suppression`, `SUPP`)

Remove ink pixels (d_B > 0.04) inside A's marker-grade footprint, minus protected real ink, blending toward paper.
- Footprint: A's chroma blurred 0.01 in, > 0.15, surviving an opening of 0.04 in (drops thin-pen sources), dilated 0.05 in.
- Protected: ink outside the footprint; ink darker than 0.95x the source density or with chroma above 0.9x the source chroma (a ghost is
  paler than its source); ink whose colour direction differs from A's (graphite, other pens); dark ink (d > 0.25) within 0.06 in of
  protected ink (a stroke crossing a band stays whole).

Rules compared on Brother sheet 1 (pixels darker than 10 / 40 gl in the marker bands; raw 18,526 / 4,959; current scanpp 15,113 / 9,611):

| rule | outcome |
|---|---|
| pixelwise ceiling α·blur(d_A) | no effect; ghost/source ratio median 0.4, p99 4.9 — no usable α |
| component rule | 784 px left, but erased 1/8 overlapping blue and 1/8 neutral crops whole; tiny glyphs kept 0.25 — rejected |
| **pixel rule with protection (chosen)** | **967 / 679 px left** |

The opening came from thin-pen sources removing real stroke fragments; the chroma-strength test came from the user's red "?" dot,
whose front carries a reverse ghost of that very dot.

## 3. Metrics

Brother sheet 1 marker bands (89,563 px footprint):

| | raw | current | new |
|---|---|---|---|
| px > 10 gl | 18,526 | 15,113 | **967** |
| px > 40 gl | 4,959 | 9,611 | **679** |

- Notes retention (208,343 px outside the bars): ink recall 0.99964 current → 0.99734 new; all 482 changed px are small ghost clusters
  below the notes; no pixel of the six notes changed. Changed px outside the dilated footprint: 0.
- q75 re-encoded input: bands 13,744 → 1,347 px > 10 gl.
- The front (p1) carries pink ghosts of the back's red notes; 6,842 px removed, black print untouched.

R2's 35 directed pairs (bleed survival):

| | pooled | macro |
|---|---|---|
| current scanpp | 0.0515 | 0.0999 |
| with gate (21/35 accepted) | **0.0228** | 0.0562 |
| forced (gate bypassed) | 0.0206 | 0.0453 |

Accepted instances lose mean 3.0 % (max 8.3 %) of cleaned ink, 463 of 6.44 M dark (< 110 gl) px; 4 pairs checked by eye — removed
pixels are ghost, handwriting intact.

Synthetic overlap (Brother p2 + 24 real handwriting crops, 8 blue/red/neutral, + 20 tiny 36x36 px glyphs): retention 1.000 in every
class (0 px lost of 15,639 / 18,641 / 20,144; tiny 0 of 3,428). Limitation: faint (40 % density) same-hue blue strokes under a band keep
0.88 (min 0.505), faint tiny blue glyphs 0.61; neutral and red faint crops 0.99.

R2 faint-content retention (17 labelled regions with neighbours): 0.9437 current = 0.9437 with gate; forced onto a wrong partner 0.9434
(0.9217 before the chroma-strength test). Harm with the gate bypassed (200 random non-pair side-instances): removed ink mean 0.18 %,
max 12 %, vs 1.0 % on accepted pairs.

## 4. Speed

Warm single-thread at 8 Mpx: `ink_maps` 30 ms/page; coarse registration 6 ms/pair; `register_pair` on an accepted sheet 361 ms;
`bleed_suppression` 432 ms + apply 120 ms per side (`clean_page` 238 ms in the same run). ≈0.24 s per sheet on average (detection
0.14 s, suppression only on the 8 % accepted). Brother stack: 21 adjacent pairs 0.9 s CPU, each of 4 suppressions 0.3-1.1 s.

## 5. Pipeline design (proposed)

Separable post-pass between `clean_page` and deskew, driven by a pre-pass; pages stay independent.
- Stage 0 (pool, PDFs with 2+ pages): edge-clean each page, `ink_maps` (50 ppi, ≈0.5 MB/page). Parent runs `pair_sheets` → plan
  page → (partner, `M_ab`).
- Stage 1 (per page): `out = clean_page(...)`; if a partner exists, `s = bleed_suppression(partner_rgb, img, reg)`,
  `out = apply_suppression(out, s, paper_rgb)`; then deskew and encode. The worker decodes (and edge-cleans) the partner itself.
- Any failure → nothing applied. Safety net: discard `s` if it removes > 25 % of B's ink (observed max 8.3 %).
- A page dropped as blank is still a valid source, not a sink.

## 6. Risks

1. Thin live evidence (one strong marker sheet).
2. Faint same-hue pen strokes fully inside a band are partly removed (indistinguishable from ghost).
3. Mutual bleed only partly handled (two strong marker notes at the same mirrored spot would not resolve).
4. Out of scope: thin-pen, black-print and pencil ghosts; black-felt-tip-only bleed is not paired.
5. Coloured paper: yellow/blue leave ~2,000 px chroma residue, no false pair; pink paper gives ncc 0.03-0.19, rejected.
6. Non-adjacent real sheets (2 found) are ignored.

## Orchestrator review

All four panels inspected: on `brother_sheet1_back_1to1.png` the blue and purple bands are gone, the green, pencil and red notes are
unchanged and the "?" dot is intact; on `synthetic_overlap.png` strokes crossing bands keep their full shape; on `corpus_pairs_crops.png`
only ghost speckle changes; `false_pair_declined.png` shows identical output on the two closest negatives.

## Panels (`panels/`)

`brother_sheet1_back_1to1.png`, `corpus_pairs_crops.png`, `synthetic_overlap.png`, `false_pair_declined.png`.
