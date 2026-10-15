# scanpp full-corpus acceptance test

> Figures referenced below are not published (they show private scans).

Tester only: no source changes. Scratch dir (a scratch copy of `samples/` plus outputs, scripts, JSON results; not published)
(scripts in `scripts/`, per-file reports in `reports1/`, `pass1.json`, `pass2.json`, `validate.json`, `renderers*.json`).
All invocations used `SCANPP_NOTIFY=0`. `samples/` was never written to.

## 0. Test suite
`.venv/bin/python -m pytest -q`: **172 passed**, 12 warnings (pikepdf "missing jbig2dec decoder"), 61.8 s.

## 1/2. Corpus run (309 PDFs, one at a time, `--foreground --jobs 16 --report`, default options)
- 309 invocations, **all exit 0**. 275 processed, 34 skipped (untouched). Pass 1 total wall 398 s (+ interpreter start).
- `pikepdf` opens all 309 inputs; I found **no unparsable PDF** in samples (the odd one is the hidden file
  `2024-2025/Sec 2 Extended/.pdf`, a normal 4-page Simple Scan PDF; it processed fine and is easy to miss with a `**/*.pdf` glob).

| metric | value |
|---|---|
| processed files (scan PDFs) | 275 -> MB in 885.6, MB out 123.9 (**x0.140**, 7.1x smaller) |
| all 309 files (skipped ones left as-is) | 938.9 MB (the task's "921 MB" ~ MiB-ish; actual bytes 938.9e6) -> 177.2 MB (**x0.189**) |
| per-file out/in | min 0.016, median 0.286, max 0.911 (9.99 Art with Transformations, 3.2 Notes Honors 0.82, 9.2 Reflections 0.88: all e1) |
| pages in / out | 1917 -> 1670 (247 dropped as blank = 12.9%) |
| KB/page (kept pages) | median **67.3**, p90 100.6, p99 145.1, max 224.8, mean 71.7 |
| by scheme | e3x 1516 pp (med 67.3 / p90 95.0 / max 158.6 KB); e3 84 pp (med 49.4 / max 143.6); e1 70 pp (med 121.3 / p90 177.0 / max 224.8) |
| file scheme mix | e3x-only 209, e3-only 15, e1-only 3, e3+e3x 16, e1+e3x 32 |
| mask codec | jbig2 1599 pp, g4 1 page (see bug 4), n/a(e1) 70 |
| pages deskewed (applied) | 406 / 1670 (24%), |angle| 0.25-1.57 deg, median 0.40 |
| blank pages kept as "blank" scheme | 0 |
| wall per page (incl. ~0.5 s process start) | median 0.28 s/page-in per file, p90 0.48, max 0.77; large files (>=20 pp) 0.12 s/page; total 398.5 s / 1917 pp = 0.21 s/page-in |

Dropped blank pages: all 247 are the empty backs of duplex scans, all in `2026-2027/Sec 2/Data/*` files. Kept pages have blank_score >= 231 (next 558), dropped <= 25: wide margin.

**Skipped (34, all exit 0, reason, never modified):** 27 "has fonts" (digital PDFs), 5 "image has a mask (already MRC-encoded?)"
(`2026-2027/Sec 3/1.1..1.4 Notes*.pdf`, `Sec 2/Standard 1-G.CO.9 Rubrik.pdf`), 2 "N XObjects (need exactly 1)"
(`2024-2025/Sec 2/21.0 Notes - Trig Applications.pdf` 6 XObjects, `2025-2026/.../16 - Notes - Sec 2 - Trig Ratios.pdf` 2).
No failures.

**Second pass on the outputs:** 306 of 309 skipped and **byte-identical**. **3 were re-processed** (not identical): see bug 2.

## 3. Validation of the 275 outputs
- `pikepdf` opens all; `check_pdf_syntax()` clean for all 275 (pikepdf lacks jbig2dec so JBIG2 streams are only syntax-checked; the renderers decode them); page count equals report `pages_out` for all.
- pdftoppm and mutool at 50 dpi: every page of every file rendered, no stderr/errors. (mutool prints "warning: ICC support is not available" once per file: build property of this mutool 1.27.0, not a scanpp problem.)
- Renderer agreement, 40 random output pages (seed 7), 100 dpi, poppler / mupdf 1.27 / gs 10.06 / PDFium (pypdfium2 5.14, already in `.venv`; nothing installed). Mean-abs-diff over RGB (0-255). gs renders 1 px shorter (1098 vs 1099 rows), so I report raw (crop to min size) and best-of-+-1px-shift:

| pair | mean of means raw / aligned | max mean raw / aligned | mean p99.9 raw / aligned | pages >4 raw / aligned |
|---|---|---|---|---|
| poppler-mupdf | 2.94 / 2.30 | 9.47 / 6.7 | 124.6 / 104.2 | 12 / 5 |
| poppler-gs | 5.11 / 2.33 | 11.8 / 6.21 | 192.2 / 135.7 | 26 / 3 |
| poppler-pdfium | 1.82 / 1.75 | 5.32 / 5.31 | 93.7 / 90.0 | 2 / 1 |
| mupdf-gs | 3.22 / 2.25 | 7.85 / 5.81 | 132.9 / 101.5 | 12 / 1 |
| mupdf-pdfium | 1.47 / 1.47 | 5.10 / 5.10 | 70.1 / 70.1 | 4 / 4 |
| gs-pdfium | 4.14 / 2.60 | 9.26 / 7.46 | 165.7 / 129.9 | 21 / 6 |

  26 of 40 pages exceed 4/255 on some pair (raw); after +-1 px alignment 7 pages (idx 3, 11, 15, 16, 26, 30, 37) still do. p99.9 is huge (hundreds of levels) because thin 1-bit-mask ink strokes land on different sub-pixels per renderer. I looked at the worst page (`sheets/renderers_crop_p37.png`, 2x crops from the four renderers): same content, same colours, only anti-aliasing/stroke weight differs (gs lightest). No renderer shows missing/garbled content, inverted mask or wrong colour. Verdict: agreement is visually fine, but the 4/255 mean criterion is not met by raw pixel comparison for ink-dense pages (expected with 1-bit masks).

## 4. Visual review (all sheets in `research/acceptance/sheets/`, <= 2000 px wide)
- `contact_01..12.png`: 48 random pages (seed 11), ORIG (from `.originals/`) | OUT.
- `crops_1/2.png`: 24 random 1:1 crops at 300 dpi in the densest windows (seed 5), ORIG | OUT.
- `dropped_00_topink.png`, `dropped_01..06.png`: **all 247** dropped original pages (top-15 ink first). All are blank (ghost bleed-through only); max pixels differing >50 levels from paper = 0.008%. **No wrongly dropped page.**
- `top10_out.png`: the 10 largest output KB/page pages (162-225 KB). All are scheme `e1` (plain JPEG): colour-art/crayon page (B7 Formative 2 p85), dense coloured pen on graph paper (20.0 Graphing Quadratics p2), full-page light-blue paper (9.99 Art p4), spiral notebook pages with heavy multi-colour pen. They are large because of ink/non-white thresholds (ink-heavy or highlighter), not a bug.
- Generally good: paper to white, ink colours kept, bleed-through gone, coils removed, deskew fine, colour pages (pink/yellow/green) clean, no halos or blotches in ordinary text pages.

## Defects, ranked by severity
1. **Small images crash (16 of 103 sample jpg/png).** `scanpp image.png` exits 1 with `ValueError: operands could not be broadcast together with shapes (77,) (81,)` in `geometry._coil_extent` (np.convolve mode="same" returns len max(n,k) when the 81-tap kernel is longer than the edge run). Repro: `bin/scan-postprocess image/png false <copy of samples/2022-2023/Secondary 2 Honors/Term 4 Book Images/image15.png (228x231)> ""` (or `python -m scanpp x.png -o y.png`). Original is left untouched, but in simple-scan background mode this is only a log line (scanpp.log) + a "failed" notification. All 16 are < ~320 px on a side (image5/8-17 in "Term 4 Book Images", 8.0 Test 0004/0005/0010, Test 8 alternate-0002). Hypothesis: guard `len(run) < k` (or use mode="valid"/pad) in `_coil_extent`.
2. **Pure-JPEG (all-e1) PDF outputs are re-processed on a second run**, contradicting "scanpp output is skipped": 3 files (`2021-2022/Secondary 2 Honors/3.2 - Notes - Secondary II Honors.pdf`, `2021-2022/Secondary 2/7.99 Test - Equations and Circles and Solving Systems.pdf`, `2022-2023/Secondary 2 Honors/9.99 Project - Art with Transformations.pdf`) were rewritten and got 3-5% bigger (708,829 -> 744,504; 228,752 -> 247,339; 646,124 -> 668,946 bytes), i.e. a second JPEG generation and a second backup. Files that mix e1 + masked pages are skipped (whole file refused due to the mask page). Hypothesis: e1 pages carry no marker; add a `/Scanpp` marker or Producer check.
3. **Scanner-lid / bed area beyond the paper edge is not cleaned.** Residual dark shadow lines, grey/mint/pink bands, posterised blotches and ragged green fringes at top/bottom/side edges. Worst: all 4 pages of `3.2 - Notes - Secondary II Honors.pdf` (bottom ~15%: black line, pink line, mint/beige blotches, offset white block at left; edge_painted_frac 7%), `2024-2025/Sec 2/18.0 Notes - Factoring Review.pdf` p3 (jagged green band along the top), `4.1 Notes - Key Features and Basic Graphing.pdf` p3, `5.98 Unit Review - Graphing Quadratics.pdf` p5/p6 (green wave at bottom), `7.99 Test` p1, `9.99 Art` p4 (cyan blob at bottom). A 30 dpi strip metric flags 90 pages with >10% non-white in the bottom 8%, 39 in the top 4% (includes legitimate content). Sheets: `edgecases_1.png`, `edgecases_2.png`, `edge_bottom.png`, `z_edge18.png`. Content is not lost; cosmetic plus extra bytes (these are the largest e1 pages). Hypothesis: edge-strip painter only handles a narrow strip / coloured lid not detected as outside paper.
4. **Pale grey printed text binarised inconsistently** (partial content loss): `2026-2027/Sec 2/Review - The Case of Ace - Stations.pdf` p6 footer "Fill in the blanks together ... Station 6 of your recording sheet": middle words rendered black/bold, the ends ("Fill in", "of your recording sheet") nearly gone (`z_faint44.png`, `contact_12.png`). Also faint pencil on pink paper partly lost/lightened (`B7 Formative 2` out p34 top-right "I'm trying the hardest ..." in `contact_11.png`). Hypothesis: threshold d >= clip(0.35*local_max, 20, 88) cuts very faint strokes while a darker neighbour in the 7x7 window keeps them.
5. **Faint printed graph-paper grids become darker and ragged/dashed; pencil gets a mauve cast and thicker strokes** (`crops_1.png` crops 9, 1; `crops_2.png` crops 12, 16, 22, 19, 23). Cosmetic; grid is more prominent than the original, some grid lines dotted.
6. **One page falls back from JBIG2 to CCITT G4**: `2021-2022/Secondary 2/7.1 Notes - Secondary II - Equation of a Circle.pdf` p3, reproducible 3/3 (`jbig2 failed (3): Error in pixReadStreamPnm: read error in 4`, leptonica can't read the PBM). Output is valid (all renderers fine), just larger masks. Hypothesis: mask content/size produces a short/odd PBM for jbig2enc.
7. **Photo-like image input is tonally altered** (`images_orig_vs_out.png`): `Term 4 Book Images/image22.png` (RGBA grayscale photo of a ship, transparent border) -> 8-bit L, alpha dropped (border becomes white), highlights/sky blown to white, contrast strongly increased. 'Venn diagram' JPEG (`Probability-0001.jpg`) -> grey L with slightly mottled mid-grey fill. Scanned-page JPEGs (page2.jpg, 7.99...-3.jpg) look good, coils removed. Expected for a scan cleaner, but worth knowing for photos/PNGs with alpha.
8. Minor: tiny stray specks left (e.g. contact_06 OUT 20, a speck at the top-right; contact_02 OUT 5 faint remnant of pink-paper ghost writing).

## 5. simple-scan protocol (real wrapper `bin/scan-postprocess application/pdf false <file> ""`, stdout/stderr piped, `communicate()` waits for exit + pipe EOF)
`SCANPP_NOTIFY=0` is honoured: `notify.py` `send_notification()` returns False and logs "notifications disabled; not sending" when `SCANPP_NOTIFY == "0"` (unless an explicit `SCANPP_NOTIFY_CMD` is set, or under pytest); confirmed in `~/.cache/scanpp/scanpp.log` for all 3 background jobs. No `notify-send` ever ran. The worker inherits the env.

| file | size | return latency | exit | background worker finished | output | backup `.originals/` |
|---|---|---|---|---|---|---|
| scanA (2 pp) | 364 KB | **0.175 s** | 0, stdout 0 B | +0.4 s | 176 KB, valid | byte-identical |
| scanB (Sec 2 B8 Formative 2, 124 pp) | 38.6 MB | **0.273 s** | 0 | +14.6 s (93 pp, 31 blank dropped, 38.6 -> 6.6 MB) | valid, renders OK | byte-identical |
| scanC (5 pp) | 1.06 MB | **0.203 s** | 0 | +15.4 s (queued behind B by the worker lock, 0.9 s own wall) | valid | byte-identical |
| email path `.../simple-scan-AB12CD/scan.pdf` (4 pp) | 996 KB | **0.978 s** (foreground; file already replaced on return, 431,568 B) | 0 | n/a | valid | byte-identical, `.originals` created inside the simple-scan-XXXXXX dir |

The input file's mtime is unchanged at return in background mode (processing happens later) and changes when the worker replaces it. Serialisation via the worker lock works (C waited 14.3 s for B). stderr carries a few hundred bytes of log lines and closes immediately.

Image inputs via the protocol (`image/jpeg`, `image/png`), return latency 0.14-0.17 s each, worker done in 0.2-0.5 s: 4 of 5 OK (page2.jpg 211->249 KB, 7.99-3.jpg 171->167 KB, Probability-0001.jpg 69->32 KB as L, image22.png 183->108 KB as L); outputs valid, same size/dpi, backups exist. The 5th (`image15.png`, 228x231) FAILED (bug 1, original untouched, backup exists).

## Six most informative sheets (`research/acceptance/sheets/`)
1. `contact_04.png` (kid colour drawing page, bleed-through gone, spiral pages, graph grids)
2. `crops_2.png` (1:1 colour/pencil/grid fidelity)
3. `edge_bottom.png` (3.2 Notes: lid/bed area defect) with `edgecases_1.png`
4. `z_faint44.png` (pale text partially lost)
5. `top10_out.png` (largest KB/page outputs)
6. `dropped_00_topink.png` (dropped pages all blank), plus `images_orig_vs_out.png` and `renderers_crop_p37.png`.

---

# Final run (second scratch dir)

Same procedure as above (copies only, `SCANPP_NOTIFY=0`, no source changes). "Before" = the first run, before these fixes.

## Results vs. first run
| metric | before | final |
|---|---|---|
| test suite | 172 passed | **240 passed** (96 s) |
| PDFs: exit 0 / processed / skipped | 309 / 275 / 34 | 309 / 275 / 34 (same skips and reasons) |
| 2nd pass on outputs | 306 skipped identical, **3 re-processed** | **309 skipped, byte-identical** (marker fix works) |
| processed MB in -> out | 885.6 -> 123.9 (x0.140) | 885.6 -> 122.6 (x0.138) |
| all 309 files out | 177.2 MB | 176.0 MB |
| KB/page median / p90 / p99 / max / mean | 67.3 / 100.6 / 145.1 / 224.8 / 71.7 | 66.4 / 99.9 / 142.2 / 221.0 / 70.9 |
| schemes (kept pp) | e3x 1516, e3 84, e1 70 | e3x 1524, e3 88, e1 58 |
| pages in / out / dropped | 1917 / 1670 / 247 | 1917 / 1670 / 247 (identical) |
| deskewed | 406 | 406 |
| G4 fallbacks | 1 page (7.1 Equation of a Circle p3) | **0** |
| validation (pikepdf, check_pdf_syntax, page count, pdftoppm+mutool 50 dpi) | clean | clean (275/275) |
| image files (103): failures | 16 crash | **0 crash** (all 103 OK; small ones are resampled to the 200 dpi default, sizes shrink by design) |
| renderers, same 40 pages: pages >4/255 raw (6 pairs) | 12/26/2/12/4/21 | 12/27/2/12/5/21; aligned: 8 pages flagged (was 7); essentially unchanged |
| wall per page-in (median / p90) | 0.283 / 0.478 s | **0.394 / 0.720 s** |
| total pass-1 wall | 398 s | **616 s (+54%)** |
| 124-page file (Formative 2 B8), controlled A/B same machine, 2 runs each | 14.3 s (source of the first run) | **25.5 s (+78%)** |
| protocol return latency (bg) | 0.175 / 0.273 / 0.203 s | 0.17 / 0.27 / 0.22 s |
| email path (fg) | 0.98 s | 1.42 s (work itself slower) |
| big-file worker time | 14.6 s | 25.9 s |

Protocol re-check: backups byte-identical, outputs valid, email path replaced in the foreground, 8 "notifications disabled" log lines, no notify-send. Image protocol: 5/5 OK now (image15.png 228x231 -> 152x154 @200 dpi, valid).

## Defect status
| # | defect | status |
|---|---|---|
| 1 | small-image crash (16/103) | **fixed** (103/103 succeed) |
| 2 | all-e1 PDFs re-processed | **fixed** (309/309 second-pass skips, identical) |
| 3a | 3.2 Notes (Honors) bottom lid/bed junk, all 4 pages | **fixed** mostly: area now pure white; a thin grey paper-edge line remains on p1 (`final_defects_1.png`) |
| 3b | 18.0 p3 green top band | **fixed** (faint lavender tint along the right edge remains) |
| 3c | 4.1 p3 (pink top edge, coil) | **fixed** |
| 3d | 5.98 Unit Review p5 green wave at bottom | **fixed** |
| 3e | 9.99 blue-paper p4 cyan blob at bottom | **unchanged** (blob still there; a few dots after "Yes" in the last table row) |
| 3f | spiral coil remnants (4.1 Honors p3, 10.1 HW p3, Parabolas p2, 10.1 Notes p1) | improved (`final_onlost_1.png`; these were the biggest "ink lost" pages vs old and it is coil/edge junk) |
| 4 | Ace p6 footer pale text | **improved**: whole footer now consistent and readable, but the right end is still cut ("...your recordin", "sheet." lost) |
| 4b | B7 Formative 2 out p34 (orig p46) faint pencil on pink | **improved** ("I'm trying / my hardest but" now complete and as faint as the original) |
| 5a | graph-paper grids heavier/dashed (crops 9, 12, 16, 22) | **unchanged** |
| 5b | pencil on yellow (B-series p43 "143", Data p109) | **improved**: now neutral grey-brown instead of mauve; pencil on pink still has a greenish-grey cast (fresh crops 11, 14, 17, 18) |
| 6 | JBIG2 -> G4 fallback | **fixed** (0 G4 pages) |
| 7 | photo PNG (Titanic, RGBA): L output, highlights blown, alpha -> white | **unchanged** (Venn JPEG fill still mottled) (`final_images_orig_vs_out.png`) |
| 8 | stray specks | unchanged/minor |

## New problems
1. **Speed regression: +54% corpus wall, +78% on large files** (14.3 s -> 25.5 s for 124 pages, A/B on the same machine). Per-page 0.28 -> 0.39 s median. Output and rendering are fine, only slower. This is the only clear regression found.
2. Checked for, and **not found**: edge cleanup painting content (ink "lost" vs old is coil/lid removal only, all 1670 pages diffed old vs final at 50 dpi: top-8 pages inspected, none lose content), line support resurrecting bleed-through or speckle (24 fresh seed-23 pages at 1:1 plus the gain-ranked pages show only genuinely faint original pencil/ink that is now kept, e.g. B7 Formative 3 p85 yellow, no ghost text from the backs), coloured ink turning grey (blue/green/purple/pink pens identical or better in all 24 crops). The coloured-paper pencil greenish cast and the 9.99 blue-paper bottom blob are carried over, not new.
3. Minor: Ace footer still clipped at the right end; thin edge line below text on 3.2 Notes p1.

## Final sheets (`research/acceptance/sheets/final_*`)
- `final_contact_01..12.png` (same 48 pages as before, ORIG | FINAL), `final_crops_1/2.png` (same 24 crops, ORIG | FINAL).
- `final_defects_1/2/3.png`: ORIG | OLD | FINAL for every flagged defect.
- `final_onlost_1/2.png`: pages where final differs most from the old output (checked for painted content).
- `final_fresh_1/2.png`: 24 fresh pages (seed 23), 1:1 at 300 dpi, ORIG | OLD | FINAL.
- `final_images_orig_vs_out.png`.

## Run 3 (after duplex)

Tester agent, no source changes; scratch dir 3 (per-file reports `reports_auto/`, `reports_off/`, panels in `panels/`).
Every PDF was run twice, `--duplex auto` (default) and `--duplex off`, one file per invocation, `--foreground --jobs 16 --report`, `SCANPP_NOTIFY=0`, on copies.

| check | result | verdict |
|---|---|---|
| tests | 275 passed | PASS |
| corpus | 309/309 exit 0; 275 processed, same 34 skips; 885.6 -> 122.3 MB (x0.138); KB/page median 66.2, p90 98.5, max 221.0; jbig2 1617 pp, 0 G4; 406 deskewed | PASS |
| blank drops | 242 (was 247): 5 dusty labelled blanks with scores 17.9-25.0 in `2026-2027/Sec 2/Data/` are now kept because the default threshold went to 15; highest dropped 12.89, lowest kept 17.94; identical with duplex off | PASS (explained) |
| duplex stats | 127 accepted pairs in 86 files; 190 pages changed (exactly the pages whose image streams differ from off); removed px median 7,721, p90 24,346, max 57,872; 0 cap discards, 0 errors | PASS |
| A/B review | 190 differing pages, 0.0-0.3 % of page pixels at 100 dpi; 20 pages reviewed (12 largest + 8 random) with the mirrored partner crop: every change is ghost removal or 1-2 px fringe on strokes; no content loss | PASS (sample) |
| second pass | 309/309 skipped, byte-identical | PASS |
| validation | pikepdf, `pdftoppm`, `mutool draw` render every page of all 275 outputs; page counts match | PASS |
| timing | corpus 567 s auto vs 424 s off (+34 %, many 2-6 page files); 124-page file 18.1 s vs 15.5 s (+16.6 %) | info |
| Brother stack | 22 -> 18 pages, 28.4 -> 1.49 MB; 1 pair (p1-2, ncc 0.61); p2 marker-band ghosts removed, all notes unchanged | PASS |

Orchestrator review: checked `sheet_crops_1.png` and zoomed `3.4 Notes - Building Polynomials` p4 (largest grid-paper change): the
removed pixels are pale blue mirrored blobs between grid lines; the grid, curve and red dots are intact. Minor effect: ink of a colour
similar to dense partner ink loses 1-2 px of edge fringe (e.g. `6.0 Triangle Congruency` p3 "Not Congruent"); strokes keep colour and weight.
