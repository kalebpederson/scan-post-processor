# R3: PDF encoding study (final cleaner = `clean_final/`, snapshot of `research/r2_cleaning/clean.py`)

> Figures referenced below are not published (they show private scans).

Everything is in `research/r3_encoding/`. Re-run with another cleaner: `CLEANER_DIR=/abs/dir_with_clean.py ./run_all.sh` (results go to `work/<dirname>/`). `clean_snapshot/` is the earlier R2 snapshot (used for the first pass; its numbers are not reported except where noted). Full tables: `work/clean_final/res/tables.md`; Pareto plots `pareto_ssim.png`, `pareto_rec300.png`, `pareto_epson.png`.

## Method
- Data: all 66 manifest pages (`single26.png` has bogus 833 dpi metadata; treated as a 300 dpi page) + Epson 300 dpi scan (rotated 90 deg CW as in R1). 150/200 dpi pages are upsampled to 300 (INTER_CUBIC) before cleaning, so their native resolution caps quality. Cleaning: `clean_page(rgb, 300, paper="flatten")` defaults. Legibility and the "main" numbers lean on the 22 native-300 pages with >=20 reference words (+ Epson).
- Candidates: `encoders.py` (`enc_jpeg`, `enc_palette`, `enc_mrc`, `enc_stencils`), registry `configs.py` (~75 configs), PDFs built with pikepdf (`build_pdf`). Page MediaBox = px*72/300, so a 300 dpi render is pixel-aligned with the cleaned reference. bytes/page = single-page PDF size minus the 625-byte empty-page container. "124-page doc" = mean * 124.
- Legibility: R1's OCR round trip, code copied into `ocrlib.py` (tesseract word recall). The decoded PDF is rendered at 300 dpi (poppler `pdftoppm`; also PDFium for the finalists) and OCR'd. Epson: R1 consensus reference (388 words), positional recall after the bilinear upsample to the 600 grid, psm 11, light-halftone class (n=270) separately. Corpus: multiset recall vs native OCR (psm 3, conf>=90) of the ORIGINAL scan.
- Fidelity (vs cleaned 300 ppi reference, poppler render): SSIM on luminance; CIEDE2000 mean over ink pixels with ref C*>25; ink-loss = ref pixels rendered white (>=245 min-channel) for solid (min<128) and faint (128..199) ink; speckle = exactly-white ref pixels rendered <245.
- Compat: poppler, MuPDF (`mutool draw`), Ghostscript png16m, PDFium (pypdfium2), and pdf.js 6.4 (node + @napi-rs/canvas, `pdfjs/render.mjs`, needs `wasmUrl` for its JBIG2 decoder) on 5 pages (single13, pair09_front, pair01_front, pair07_front, single26) x 17 configs; pass = all 5 render, SSIM vs ref consistent, max pairwise mean-abs-diff (any pair, incl. pdf.js) < 5/255; `pikepdf.check_pdf_syntax()` clean.
- Classes (cleaned pages): blank 0/66 (rule: <0.02% non-white pixels; blank page = empty content stream, ~1 KB; tested only synthetically in code path), no-colour 3/66 (4.5%: pair07_front, pair07_back, image02), colour 63/66 (+Epson, colour). Rule: <1500 saturated ink px or <0.5% of ink px with (max-min)>40.

## Key design facts found
1. **Mask threshold matters.** A global ink threshold d=255-min(rgb) >= 96 drops half of the light halftone text (Epson light recall 0.26). Adaptive threshold `clip(0.35*local_max_7x7(d), 20, 88)` matches the unthresholded reference (and keeps pale highlighter under 88). Used for E3/E4 (`ADAPT`).
2. **The colour layer, not the mask, is most of the bytes.** Mask (JBIG2 300 ppi) is only ~21-25 KB/page; naive nearest-fill colour layer was 120 KB. Smooth push-pull inpainting of non-ink cells (`pushpull`) + 75 ppi q40 4:2:0 gives ~30-45 KB.
3. JBIG2 generic (`jbig2 -p -d -G`, no `-s`) is ~13% smaller than CCITT G4; both render identically everywhere.
4. Highlighter: detector (pale saturated wide regions, opening >= 25 px) moves highlighter into a separate 75 ppi background JPEG and measures ink relative to it (`hl=True, hl_w=25`); the 9 px version falsely fires on graph grids (blotches) - rejected; hl25 did not fire on graph/handwriting pages tested (pair09_front, single13, pair03_back) and fires on highlighter pages (single09, single26, pair06_front, pair04_back); mean bytes unchanged (67.5 vs 67.5 KB).

## Results (final cleaner; 66 pages unless noted)
| cfg | KB med | mean | p90 | 124-pg MB | OCR300 recall poppler / PDFium | Epson pos poppler | Epson pos / light PDFium | SSIM | dE00 | ink-loss solid/faint | speckle | enc s | compat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **E0** current (magick, JPEG q70 2544 px + SMask) | 493 | 492 | 632 | 61.0 | 0.956 / - | 0.644 | - | 0.923 | 15.4 | 0.071/0.329 | 0.040 | 1.8 | (not tested) |
| E0 without SMask (`-alpha off`) | 360 | 360 | 500 | 44.6 | 0.956 | 0.644 | - | 0.923 | 15.4 | same | same | - | - |
| E1 JPEG 150 ppi q40 4:2:0 | 105 | 108 | 141 | 13.4 | 0.953 / 0.967 | 0.778 | 0.796 / 0.852 | 0.969 | 7.7 | 0/0 | 0.026 | 0.03 | PASS (2.3) |
| E1 JPEG 150 ppi q55 4:2:0 | 125 | 127 | 167 | 15.7 | 0.965 / 0.959 | 0.799 | 0.789 / 0.859 | 0.972 | 7.1 | 0/0 | 0.022 | - | - |
| E1 JPEG 200 ppi q55 4:2:0 | 183 | 186 | 247 | 23.1 | 0.933 / 0.943 | 0.840 | 0.851 / 0.907 | 0.979 | 5.7 | 0/0 | 0.015 | 0.05 | PASS |
| E1 JPEG 200 ppi q85 4:2:0 | 284 | 294 | 395 | 36.5 | 0.972 / 0.965 | 0.832 | 0.835 / 0.885 | 0.983 | 4.9 | 0/0 | 0.009 | - | - |
| E2 palette k=8 150 ppi (k-means) | 105 | 105 | 144 | 13.0 | 0.961 / 0.959 | 0.827 | 0.820 / 0.896 | 0.972 | 13.7 | 0/0.004 | 0.007 | 0.79 | PASS (2.4) |
| E2 palette k=16 200 ppi (k-means) | 206 | 204 | 274 | 25.3 | 0.961 / 0.961 | 0.838 | 0.845 / 0.893 | 0.979 | 8.8 | 0/0.004 | 0.006 | 1.41 | PASS (2.4) |
| E2 palette k=4 150 ppi (k-means) | 60 | 61 | 79 | 7.6 | 0.955 / - | 0.817 | - | 0.965 | 20.7 | 0/0.005 | 0.007 | - | - |
| E3 mask300 JBIG2 + colour 100 ppi q55 ("E3_base") | 98 | 99 | 131 | 12.3 | 0.956 / 0.957 | 0.851 | 0.851 / 0.896 | 0.967 | 7.8 | 0.004/0.096 | 0.002 | 0.57 | PASS (3.1) |
| **E3 mask300 JBIG2 + colour 75 ppi q40 ("E3_c75_q40")** | 66 | 67 | 88 | 8.3 | 0.946 / 0.958 | 0.845 | 0.853 / 0.896 | 0.960 | 9.5 | 0.004/0.096 | 0.002 | 0.54 | PASS (2.8) |
| E3 same + highlighter layer (`_hl25`) | 66 | 67 | 88 | 8.4 | 0.951 / 0.958 | 0.845 | - | 0.961 | 9.5 | 0.004/0.096 | 0.002 | 1.1 | - |
| E3 colour 50 ppi q40 | 50 | 51 | 67 | 6.3 | 0.938 / 0.928 | 0.853 | 0.830 / 0.863 | 0.955 | 10.8 | 0.004/0.094 | 0.002 | - | - |
| E3 mask 200 ppi, colour 75 q40 | 56 | 57 | 75 | 7.0 | 0.935 / 0.959 | 0.796 | 0.809 / 0.863 | 0.965 | 9.1 | 0/0.040 | 0.003 | - | PASS (2.6) |
| E3 mask 400 ppi (upsampled from 300) | 109 | 111 | 147 | 13.8 | 0.949 | 0.838 | - | 0.948 | 9.2 | 0.041/0.27 | 0.002 | - | PASS (4.0) |
| E3 with CCITT G4 mask (colour 100 q55) | 107 | 113 | 146 | 14.0 | 0.956 | 0.851 | - | 0.967 | 7.8 | same | same | - | PASS (3.1) |
| E4 stencils K=2 colours | 61 | 61 | 86 | 7.6 | 0.882 / 0.882 | 0.827 | 0.835 / 0.919 | 0.936 | 22.1 | 0/0.031 | 0.002 | 0.71 | PASS (1.7) |
| E4 stencils K=4 | 100 | 104 | 149 | 13.0 | 0.891 / 0.891 | 0.845 | 0.840 / 0.922 | 0.944 | 17.9 | 0/0.012 | 0.003 | - | - |
| E4 stencils K=7 | 137 | 142 | 198 | 17.6 | 0.899 / 0.899 | 0.848 | 0.832 / 0.896 | 0.949 | 14.3 | 0/0.008 | 0.003 | 1.30 | PASS (1.7) |

(Compat column: max pairwise mean-abs-diff over the 5 renderers incl. pdf.js, /255. Epson/OCR noise: Epson pos recall CI ~ +-0.04, OCR300 page-mean ~ +-0.02, so everything >= ~0.80 / >= ~0.93 is "equal within noise" except E4 on corpus (0.88-0.90 is a real loss; stencil colours wipe out some thin pencil/colour text) and E1 150 on Epson light text (0.85 vs 0.90 at 200+).) Oracle-free per-class bytes (mean KB colour / gray pages): E0 497/377, E1_j150_q40 109/81, E2_p150_k8 107/73, **E3_c75_q40 68/44**, E3_c50_q40 52/33, E4_s2 62/50. Grayscale variants on the 3 no-colour pages (KB mean): E2g 4-level uniform 150 ppi 22, E2g 4-level 200 ppi 33, **E3g (gray colour layer) 43**, E1g_j150_q40 75; all OCR300 >= 0.97; the gray 4-level Flate pages are the smallest but have hard-quantised antialiasing (speckle 0.0006-0.012).

### Findings per candidate
- **E0**: the command as literally given (ImageMagick 7.1.2) writes RunLength (2.4 MB/page, 386 MB for 124 pp). With `-compress jpeg` it gives 492 KB/page mean (matches the user's ~468 KB). It does not resample: pixels stay at 2544x3295, `-density 200 -page Letter` only relabels the page as 220x285 pt (~833 ppi, same as the 833 dpi in the user's file metadata), and an 8-bit SMask (~27% of the bytes) is added; `-alpha off` saves 27%. Its level adjustment crushes light text: Epson recall 0.644 vs 0.78-0.85. Time 1.8-2.5 s/page (Ghostscript rasterise dominates).
- **E1**: robust and everywhere-identical; ringing "speckle" 1-2.6% at 150 ppi q40 (mosquito noise visible at 200% zoom near thin colour lines, not at 100%); 4:4:4 costs +20-40% for no legibility gain (ΔE 5.4 vs 7.7 only). 150 ppi hurts Epson light text (consistent with R1), 200 ppi fixes it (+75% bytes).
- **E2 palette**: median-cut vs k-means: k-means wins on colour (dE 13.7 vs 17.7 at k=8; mean OCR equal); both are bad at k<=8 for colour (light blue pen becomes gray, pale highlighter becomes gray-blue: see panels p2, p4, p6). k=16 is okay but 2-3x larger than E3.
- **E3**: smallest at equal legibility, sharpest tiny text (300 ppi 1-bit), but: (a) ink-loss "faint" 9.6% = antialiasing fringe/pencil lighter than threshold dropped (pencil looks slightly lighter and thinner, still legible: panels p3, p3b), (b) 75 ppi colour layer gives slight stair-stepped colour at pen edges at 200% zoom and colour shifts (dE 9.5 vs 7.7 for JPEG), (c) poppler renders explicit /Mask with edge antialiasing, PDFium/MuPDF/Ghostscript/pdf.js hard-edged (pairwise MAD up to 3.1/255, p99.9 172, inherent for 1-bit masks; no structural difference).
- **E4**: sharp and compact for K=2, but colours collapse (blue to gray, orange to pink, grid lines purple-gray), pencil darkens (panels p2, p3b, p5): reject for colour work.
- E3 variants: threshold 128 global loses light text (Epson 0.70, faint loss 70%); 400 ppi mask is worse (upsampled blobs), 200 ppi mask is -15% bytes but Epson 0.80. Colour q70/4:4:4/150 ppi cost +50-130% for +0.5 dE.

## Cleaned vs uncleaned Epson (final cleaner; same encoder and density; `epson_clean_vs_raw.py`)
Epson positional recall (light class) poppler / PDFium:
| encoder | uncleaned input | cleaned input |
|---|---|---|
| E1 150 q55 | 0.796 (0.870) / 0.820 (0.907), 167 KB | 0.799 (0.874) / 0.789 (0.859), 222 KB |
| E1 200 q70 | 0.817 (0.874) / 0.817 (0.874), 326 KB | 0.827 (0.874) / 0.840 (0.904), 416 KB |
| E1 200 q85 | 0.825 (0.881) / 0.838 (0.885), 489 KB | 0.832 (0.885) / 0.835 (0.885), 578 KB |
| E2 200 k16 | 0.807 (0.867) / 0.822 (0.874), 625 KB | 0.838 (0.885) / 0.845 (0.893), 359 KB |
Cleaning is neutral for legibility (|diff| <= 0.03, inside noise; slightly positive for E2/E1-200) and is required for the mask schemes (they need white paper). It increases JPEG bytes on this page (+25-33%; the cleaner boosts faint pencil/rule lines, i.e. real content) but shrinks palette bytes. With the earlier cleaner (clean_snapshot) cleaned Epson recall was 0.67 vs 0.88 uncleaned (light text erased), so the final cleaner's boost fixed that.

## Haze (faint gray left by graphite smudges)
`haze.py`: setting faint pixels (1<=d<=25) not adjacent to real ink to white (0.0-2.6% of pixels per page; 8 worst pages 1.8-2.6%) changes bytes by: E1 150 q40 -0.05% mean (max -0.3%), E1 200 q55 -0.16%, **E3_c75_q40 -0.46% (max -1.3%)**, E4_s7 -1.3% (noisy, +-9%), E2 k8 -1.2% (noisy, +-12% from k-means palette instability). So haze costs essentially no bytes in the finalists; speckle metric (ref-white only) is unaffected by it because haze is non-white in the reference. The pencil-heavy page pair03_back: E3_c75_q40 59 KB (cf. page mean 67 KB), E1_j150_q40 100 KB; Epson (dense pencil + ruled paper) 121 KB vs 183 KB.

## Compatibility (5 pages x 17 configs, 5 engines)
All 85 PDF/engine combos rendered; `pikepdf.check_pdf_syntax()` empty for all 6 sample PDFs and every tested config. Pairwise MAD (max over the 5 engines, /255): E1 2.3-2.4 (p99.9 81-83), E2 2.4 (101-105), E3 2.6-3.1 (145-172; E3 400 ppi mask 4.0), E4 1.7 (111-125). Per-engine SSIM spread vs reference <= 0.01 (poppler lowest for /Mask, Ghostscript lowest for upscaled JPEG, all within 0.01). pdf.js initially produced blank output for JBIG2/G4 masks in node: it needs its `jbig2.wasm` (`wasmUrl`), real browser builds ship it; with it the output is identical to the others. Not tested: Acrobat, Apple Preview/iOS, Edge (PDFium-based as Chrome), Android PDFium = PDFium. Standard constructs only (Image `/Mask` -> ImageMask stencil with JBIG2Decode generic, DCTDecode baseline, Flate+PNG predictor, ImageMask+fill colour, Indexed); Preview/Acrobat are the residual risk for JBIG2 /Mask on a low-res image with different resolution: if a viewer misbehaves, `E3_g4` (CCITT G4 mask, +14% bytes) is the drop-in alternative.

## Panels (`panels/`, <=2000 px; rows: cleaned ref, E3_c75_q40, E3_c75_q40_hl25, E1_j150_q40_420, E2_p150_k8_km, E4_s4, E1_j200_q55_420; columns: PDFium render at 110 ppi 100%, 110 ppi 200%, 160 ppi 100%)
- `p1_epson_tiny_halftone.png` (Epson 6 pt light text): E3/E4 crisp, JPEG 150/200 blurry.
- `p2_handwritten_math_exponents.png`: E3 and JPEG good; E2 k8 washed-out blue, E4 loses green/orange.
- `p3_pencil_epson.png`, `p3b_pencil_pair03back.png`: E3 pencil slightly lighter/thinner, E4 pencil too dark, JPEG faithful.
- `p4_highlighter_over_text.png`: plain E3 darkens/fringes ink on highlighter; `_hl25` fixes it; E2 turns highlighter gray, E4 turns ink magenta.
- `p5_graph_grid.png`: E3 ok (slightly mottled grid), E4 purple-gray grid, E2 ok.
- `p6_thin_colored_lines.png`: JPEG ringing barely visible, E3 slightly stair-stepped light-blue/cyan at 200%, E2/E4 lose blue.
(Panels were generated with the commands at the end of this file.)

## Recommendation
**E3 with JBIG2-generic 300 ppi ink mask + 75 ppi JPEG q40 4:2:0 colour layer, highlighter layer (hl_w=25)**: ~67 KB/page mean (median 66, p90 88; 8.4 MB per 124 pages, vs E0 61 MB; 7.3x smaller) with legibility at the best level within noise (Epson 0.845 pos / 0.896 light, OCR300 0.95 vs best 0.97, E0 0.956) and the best tiny-text sharpness of any option. It is smaller than E1 150 q40 on 66 of 67 pages (median ratio 0.61), and 2.7x smaller than E1 200 q55, the cheapest JPEG that matches on light text. Cost: slightly lighter pencil, mild colour shift (dE 9.5 vs 7.7), a bit blocky colour at 200% zoom, speckle 0.2% (JPEG has 2.6%).
Per-page rule (`pageselect.py`, reference implementation; samples in `samples/`):
- blank (<0.02% non-white) -> empty content stream (~1 KB);
- no-colour (<0.5% saturated ink) -> same E3 with a DeviceGray colour layer (44 KB mean on the 3 gray pages vs 81 KB for colour JPEG 150 q40 on those pages; 4-level uniform gray Flate at 150 ppi is 22 KB but posterised; use only if bytes dominate);
- ink-heavy art pages (ink >15% of pixels or non-white >30%, e.g. single38) -> JPEG 150 ppi q40 4:2:0 (E3 was larger there: 116 vs 111 KB, SSIM 0.82 vs 0.78);
- safe fallback for any viewer/legibility concern: E1 200 ppi q55 4:2:0 baseline JPEG (186 KB mean, SSIM 0.979, every engine identical);
- if the document contains tiny (<7 pt) light text AND bytes are not critical, E3 colour layer 100 ppi q55 (99 KB) is a stronger-colour variant.
Do not use: E4 (colour/pencil damage), E2 below k=16, JPEG 4:4:4, mask ppi 400, global mask thresholds.

## PDF construction (reimplementation)
Per page: `/MediaBox [0 0 W*72/300 H*72/300]`, `/Resources << /XObject << /Fg 5 0 R >> >>`, content `q Wpt 0 0 Hpt 0 0 cm /Fg Do Q` (with a highlighter: `... /Bg Do Q q ... /Fg Do Q`).
- `/Fg`: `/Type/XObject /Subtype/Image /Width cw /Height ch /ColorSpace/DeviceRGB (DeviceGray for no-colour pages) /BitsPerComponent 8 /Filter/DCTDecode /Mask <Mk>`; cw,ch = round(W*75/300): baseline sequential Huffman-optimised JPEG (Pillow `quality=40, subsampling=2, optimize=True, progressive=False`). Pixels: darkness^3-weighted mean colour of ink pixels per 4x4 cell, empty cells filled by push-pull inpainting.
- `/Mk`: `/Type/XObject /Subtype/Image /Width W /Height H /ImageMask true /BitsPerComponent 1 /Filter/JBIG2Decode` (no DecodeParms/Globals, no `-s`), data from `jbig2 -p -d -G` of a 0/255 8-bit PNG (ink = 0), i.e. embedded generic-region stream, ink = sample 0 = painted. CCITT alternative: `/Filter/CCITTFaxDecode /DecodeParms << /K -1 /Columns W /Rows H /BlackIs1 true >>`, single strip (PIL TIFF group4, RowsPerStrip=H).
- `/Bg` (optional): small JPEG (75 ppi q60 4:2:0) of the highlighter-only layer (white elsewhere), drawn first at full page size; the ink mask is then computed as darkness relative to the upscaled Bg.
- Ink mask: ink if d >= clip(0.35*maxfilter_7x7(d), 20, 88), d = 255-min(R,G,B) on the cleaned 300 ppi image.
- Other filters used in the study: E1 `/DCTDecode`; E2 `[/Indexed /DeviceRGB k-1 <palette>]` with `/Filter/FlateDecode /DecodeParms << /Predictor 15 /Colors 1 /BitsPerComponent b /Columns w >>`, b=2/4/8 for k<=4/16/32 (own per-row PNG filter selection + zlib 9, `png_predict`); E4 one `/ImageMask` per colour painted with `r g b rg ... /Mi Do`, lightest colour first.
- Container: pikepdf `object_stream_mode=generate`, `compress_streams=True` (only content streams are re-compressed), overhead 625 B/file.

## Timing (single thread, shared machine, median of 3 pages x 2): E1 0.03-0.06 s, E3 0.54 s (1.1 s with highlighter detection), E4 0.7-1.3 s, E2 0.8-1.4 s, E0 1.8 s (+2.5 s under load) per page, cleaning 2.8 s (final cleaner).

## Commands used for panels
`panels.py <name> <page> <x0,y0,x1,y1 at 300 ppi> E3_c75_q40,E3_c75_q40_hl25,E1_j150_q40_420,E2_p150_k8_km,E4_s4,E1_j200_q55_420 "<title>"` with: p1 epson300 1500,2190,2400,2390; p2 single13 1400,2280,2300,2580; p3 epson300 2608,100,3508,300; p3b pair03_back 380,1850,1280,2150; p4 single09 650,2080,1250,2330; p5 pair09_front 150,130,750,430; p6 pair06_front 250,1000,1000,1350.

## Caveats
Blank pages: none in the corpus (path implemented, not corpus-tested). Only 3 no-colour pages. Acrobat/Preview not available here. 150/200 dpi pages carry upsampled content. OCR numbers are noisy (CI ~ +-0.04 Epson); differences < 0.03 are not significant. The old-cleaner pass (`work/clean_snapshot/`) showed the same ranking (E3_c75_q40 54 KB vs JPEG150 q40 100 KB). image01 (yellow-tinted paper) was mishandled by the old cleaner (paper tint left, masks failed there); not rechecked.


# Round 2 (colour identity, highlighter fringe, pencil weight; E3 refinements; E5 hybrid)

Superseding note: the Round-1 recommendation (E3 75 ppi 4:2:0 colour layer) is **withdrawn**. Round-2 metrics show it changes colour coding on ~11% of stroke components (hue-shift 2.8%, swaps of adjacent strokes 1.5%) and is blotchy on highlighter.

## New metrics (`metrics2.py`, `run_metrics2.py`, `aggregate2.py`; PDFium render at the 300 ppi reference grid, all 66 pages, mean over pages)
- **Colour identity**: reference ink pixels with Lab C*>25 are split per 30-degree hue bin and 8-connected into stroke components (area >= 30 px; headline uses components >= 100 px; ~600-1000 per colour page). Per component the median a*,b* of its core pixels (darkness >= 100) is compared to the render: "changed" = hue shift > 20 deg OR chroma < 0.6 x reference. Also reported: hue-shift only, area-weighted %, and **swap %** (changed component whose rendered hue matches the reference hue of a different-hue neighbouring component within 4 px = adjacent strokes took each other's colour; as % of all components).
- **Highlighter fringe**: highlighter regions detected on the reference (pale saturated, opening 25 px; 7 pages: pair04_back, pair06_front, single09, single12, single19, single26, single38). CIEDE2000 of (a) ink pixels inside the region and (b) highlighter pixels within 2 px of ink.
- **Pencil weight**: sum of (255 - gray) of render / reference over the 4 pencil boxes of `r2_cleaning/labels.json` (pair03_front x2, image04, Epson hatch; coordinates mapped to the 300 ppi/rotated grids). 1.00 = same ink weight; <1 lighter.

## Round-2 table (mean over pages; OCR/Epson from the Round-1 harness, poppler)
`work/clean_final/res/tables_r2.md` has every candidate (91 rows); the relevant ones:

| cfg | KB mean | OCR300 | Epson pos/light | SSIM | dE | colour-ID changed % (comp>=100px) | hue-shift % | area-wtd % | swap % | hl ink dE | hl near-ink dE | pencil wt (mean/min) | speckle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 492 | 0.956 | 0.644/0.678 | 0.9233 | 15.4 | 26.1 | 11.5 | 22.7 | 0.99 | 18.8 | 15.2 | 0.37/0.34 | 0.0404 |
| E1_j150_q40_420 | 108 | 0.953 | 0.778/0.848 | 0.9690 | 7.7 | 22.0 | 1.1 | 18.8 | 0.45 | 6.8 | 5.4 | 1.05/1.05 | 0.0261 |
| E1_j150_q40_444 | 130 | 0.950 | 0.776/0.841 | 0.9692 | 5.4 | 0.5 | 0.0 | 0.5 | 0.01 | 5.8 | 4.7 | 1.05/1.05 | 0.0182 |
| E1_j150_q55_420 | 127 | 0.965 | 0.799/0.874 | 0.9722 | 7.1 | 13.6 | 0.5 | 11.7 | 0.37 | 6.3 | 5.1 | 1.05/1.05 | 0.0224 |
| E1_j150_q55_444 | 152 | 0.960 | 0.807/0.878 | 0.9725 | 5.1 | 0.3 | 0.0 | 0.3 | 0.00 | 5.6 | 4.4 | 1.05/1.05 | 0.0153 |
| E1_j200_q55_420 | 186 | 0.933 | 0.840/0.900 | 0.9791 | 5.7 | 2.2 | 0.1 | 1.9 | 0.08 | 4.9 | 4.3 | 1.04/1.03 | 0.0146 |
| E1_j200_q55_444 | 227 | 0.939 | 0.843/0.900 | 0.9793 | 4.5 | 0.2 | 0.0 | 0.3 | 0.00 | 4.4 | 3.7 | 1.03/1.03 | 0.0102 |
| E2_p150_k16_km | 130 | 0.955 | 0.840/0.889 | 0.9759 | 8.8 | 14.3 | 9.0 | 11.7 | 1.04 | 12.2 | 8.0 | 1.00/1.00 | 0.0080 |
| E3_444 | 112 | 0.949 | 0.845/0.900 | 0.9667 | 6.7 | 0.2 | 0.1 | 0.2 | 0.16 | 7.0 | 9.6 | 1.00/1.00 | 0.0021 |
| E3_c75_q40_hl25 | 67 | 0.951 | 0.845/0.889 | 0.9605 | 9.5 | 10.7 | 2.6 | 9.2 | 1.47 | 9.5 | 9.0 | 1.01/1.00 | 0.0022 |
| E3r_c100_q55_dom | 103 | 0.950 | 0.822/0.863 | 0.9647 | 8.1 | 3.7 | 0.8 | 4.2 | 0.77 | 9.7 | 8.9 | 0.99/0.99 | 0.0022 |
| E3r_c100_q55_dom_dil | 114 | 0.929 | 0.753/0.815 | 0.9533 | 8.4 | 5.5 | 1.1 | 5.8 | 0.77 | 9.7 | 11.5 | 1.17/1.17 | 0.0068 |
| E3r_c100_q55_dom_lo12 | 105 | 0.949 | 0.825/0.863 | 0.9645 | 8.1 | 3.6 | 0.8 | 4.2 | 0.77 | 9.6 | 9.0 | 0.99/0.99 | 0.0023 |
| E3r_c150_q55_mean | 156 | 0.952 | 0.845/0.874 | 0.9709 | 6.6 | 1.0 | 0.3 | 0.9 | 0.27 | 7.3 | 7.5 | 0.98/0.97 | 0.0022 |
| E3x_c100_q40_444_dom | 100 | 0.938 | 0.822/0.859 | 0.9642 | 7.4 | 0.6 | 0.2 | 0.8 | 0.23 | 9.0 | 8.7 | 0.99/0.98 | 0.0022 |
| E3x_c100_q40_444_mean | 97 | 0.956 | 0.830/0.867 | 0.9657 | 7.1 | 0.6 | 0.2 | 0.8 | 0.22 | 8.2 | 8.4 | 1.00/0.99 | 0.0022 |
| E3x_c120_q40_444_mean | 120 | 0.951 | 0.820/0.859 | 0.9682 | 6.5 | 0.4 | 0.1 | 0.2 | 0.14 | 7.6 | 7.8 | 1.00/0.99 | 0.0022 |
| E3x_c150_q40_444_mean | 153 | 0.955 | 0.820/0.863 | 0.9701 | 5.9 | 0.4 | 0.1 | 0.2 | 0.08 | 7.0 | 7.4 | 0.98/0.97 | 0.0022 |
| E3x_c75_q40_444_mean | 74 | 0.943 | 0.848/0.889 | 0.9606 | 8.5 | 1.0 | 0.3 | 1.7 | 0.46 | 9.0 | 8.9 | 1.01/1.00 | 0.0022 |
| E4_s7 | 142 | 0.899 | 0.848/0.911 | 0.9487 | 14.3 | 36.6 | 24.1 | 29.4 | 1.55 | 17.7 | 18.3 | 1.35/1.34 | 0.0032 |
| E5a_q40 | 118 | 0.948 | 0.807/0.859 | 0.9581 | 7.8 | 22.0 | 1.1 | 18.8 | 0.45 | 8.0 | 9.7 | 1.05/1.05 | 0.0261 |
| E5a_q40_444 | 140 | 0.952 | 0.812/0.867 | 0.9584 | 5.5 | 0.5 | 0.0 | 0.5 | 0.01 | 7.1 | 9.0 | 1.05/1.05 | 0.0182 |
| E5a_q40_444_t3 | 141 | 0.953 | 0.809/0.870 | 0.9617 | 5.5 | 0.5 | 0.0 | 0.5 | 0.01 | 6.6 | 6.8 | 1.05/1.05 | 0.0182 |
| E5a_r200_q40_444 | 205 | 0.952 | 0.840/0.900 | 0.9648 | 4.9 | 0.3 | 0.0 | 0.3 | 0.00 | 6.2 | 8.4 | 1.04/1.03 | 0.0119 |
| E5b_q40_444 | 115 | 0.947 | 0.781/0.815 | 0.9582 | 6.2 | 0.5 | 0.0 | 1.0 | 0.03 | 8.3 | 9.2 | 1.05/1.05 | 0.0142 |
| E5b_q40_444_t3 | 118 | 0.951 | 0.727/0.737 | 0.9594 | 6.2 | 0.5 | 0.0 | 1.0 | 0.02 | 7.8 | 6.9 | 1.05/1.05 | 0.0146 |

(colour-ID columns: % of stroke components >=100 px whose colour coding changed / hue-shifted; area-wtd = same by ink area; swap = adjacent-stroke colour swaps as % of components. hl = CIEDE2000 on highlighter pages. pencil wt = rendered/reference ink weight. Compat/E0 columns not repeated.)

## Findings
1. **Root cause of the colour failures (a): 4:2:0 chroma subsampling, not (only) the 75 ppi layer.** Same encoders with 4:4:4: E1 150 q40 22.0% -> 0.5% components changed (152 KB at q55: 0.3%); E3 100 ppi q55 3.7% -> 0.2%; E3 75 ppi q40 10.8% -> 1.0% (74 KB). 4:2:0 desaturates thin pen strokes (chroma < 0.6x) even though hue is right (hue-shift 1.1% for JPEG 4:2:0). The swap pattern the reviewer saw (magenta "2" next to red "x") is the 75 ppi mean-colour cells: swaps 1.5% -> 0.2-0.5% with 4:4:4 at 100-150 ppi, 0.08% at 150 ppi 4:4:4.
2. **Colour layer refinements.** The layer was already a normalized convolution over ink pixels only (weights darkness^3, empty cells filled by push-pull), so "mean over ink only" is what we had. Tried: 100/120/150 ppi, q40/q55, 4:2:0 vs 4:4:4, and a **dominant-colour** variant (Lab k-means palette of 12, per-cell pixel-count vote, average of the winning cluster only). Dominant-colour gave no benefit (swap 0.23% vs 0.22%, OCR lower 0.938 vs 0.956 at 100 ppi q40) -> keep mean. Best size/accuracy: **E3x_c100_q40_444_mean 97 KB: colour-ID 0.6% changed, hue 0.2%, swap 0.22%**; E3x_c120 q40 444 (120 KB) 0.4%; E3x_c150 q40 444 (153 KB) 0.4%, swap 0.08%. 75 ppi 4:4:4 (74 KB) leaves 1.0%/swap 0.46% - not acceptable for "≈0".
3. **Pencil (c).** Pencil-weight ratio of E3 is 1.00 (c75: 1.01, c100/150: 0.98-1.00): E3 does not lose pencil ink weight; the "lighter/thinner" impression comes from the mask (pencil pixels below the adaptive threshold are dropped while the rest are drawn at full colour - fewer mid-gray pixels). Lower threshold (lo 12) changed nothing measurable (0.99); **dilating the mask makes pencil too heavy (1.11-1.17), and hurts Epson light text (0.75-0.80) and SSIM** - rejected. JPEG renders pencil at 1.05 (the cleaner boosts it), E4 1.35-1.5 (too dark), palette 1.0, E0 0.37 (!). Stair-stepping at 200% zoom is the 100 ppi colour grid; 150 ppi removes it but costs +56 KB.
4. **Highlighter blotches (b) are inherent to a binary mask over a highlighter**: E3 ink dE 8.2 / near-ink 8.4 (100 ppi) and 7.0/7.4 (150 ppi) vs JPEG 150 q55 4:4:4 5.6/4.4 and JPEG 200 4.4/3.7 (hl layer does not fix: 9.5/9.0 at 75 ppi). Recommendation therefore routes highlighter pages to JPEG.
5. **E5 hybrid** (`enc_hybrid`: 150 ppi JPEG of the whole cleaned page + 300 ppi JBIG2-generic ImageMask painted with the median text colour (~(42,42,48)), mask = dark NEUTRAL ink only: chroma <= 40, stroke-component core mean darkness >= P=170, components ringed by saturated ink rejected (dark pen cores), edge threshold 0.55 x local max clipped 80..150; E5b first replaces the stencil pixels (dilated 1 px) in the JPEG by push-pull background):
   - first versions turned the dark core of a green stroke black and made text too bold; fixed by the ringed-by-colour rejection and the higher edge threshold (E5 `_t3`).
   - E5a 4:4:4 q40: 140 KB, colour-ID 0.5%, swap 0.01%, Epson 0.81/0.87, OCR 0.952, pencil weight 1.05 (stays in the JPEG; P=150 pulls pencil into the mask and makes highlighter worse, P=190 lowers Epson), text visibly crisper than JPEG at 200% zoom with no visible halo or misregistration (panels), but **SSIM 0.962 < E1 0.9725** (1-bit text edges vs antialiased reference), hl near-ink dE 6.8 vs 4.4, and 140 KB is not smaller than E1 150 q55 4:4:4 (152 KB). 4:2:0 versions fail colour identity (22%).
   - E5b 4:4:4 q40: 115-118 KB (-18% vs E5a) but Epson light text drops to 0.73-0.78 (the carve-out removes JPEG context around the halftone text) -> rejected.
   - E5a at 200 ppi q40 444: 205 KB, Epson 0.84/0.90: equals E1 200 (227 KB) minus 10%.
   Verdict: E5 buys sharper black text but not bytes or fidelity; it is only attractive if black-text sharpness at 100% zoom is judged more important than the 0.01 SSIM; not recommended.
6. **E1 4:4:4 is the robust baseline**: E1 150 q55 4:4:4 152 KB (colour 0.3%, hl 5.6/4.4, SSIM 0.9725, Epson 0.807/0.878, OCR 0.960); E1 200 q55 4:4:4 227 KB (0.2%, Epson 0.843/0.900, SSIM 0.979). The 150 ppi version is a little soft on 6 pt light text at 160 ppi viewing (panel p1), exactly what R1 predicted. Palette (E2) stays unusable below k=32 (colour-ID 14-95%).

## Compatibility (E5 and the new E3, 5 renderers: poppler, MuPDF, Ghostscript, PDFium, pdf.js; 5 pages; `compat_round2.json`)
All render correctly; `check_pdf_syntax` clean. Max pairwise mean-abs-diff / p99.9 (/255): E5a 4:4:4 q40 2.28 / 88, E5a `_t3` 2.28 / 87, E5b `_t3` 2.23 / 107, E3x c100 444 3.12 / 167, E3x c150 444 2.80 / 164, E1 150 q55 444 2.36 / 86, E1 200 q55 444 2.38 / 79. Mean SSIM vs reference per renderer: E5a `_t3` poppler .968 mupdf .970 gs .960 pdfium .968 pdf.js .960; E3x c100: .970 .972 .972 .972 .971. The E3/E5 mask edges differ only by antialiasing (poppler smooths /Mask edges), no structural differences. E5 uses a plain full-page ImageMask (no /Mask on a low-res image), i.e. the least exotic JBIG2 use.

## Panels (round 2; `panels/r2_*.png`; rows: cleaned ref, E1_j150_q55_444, E3x_c100_q40_444_mean, E5a_q40_444_t3, E5b_q40_444_t3, E1_j200_q55_420; columns 110 ppi 100%, 110 ppi 200%, 160 ppi 100%; round-1 panels moved to `panels/round1/`)
`r2_p1_epson_tiny_halftone`, `r2_p2_handwritten_math_exponents`, `r2_p3b_pencil_pair03back`, `r2_p4_highlighter_over_text`, `r2_p5_graph_grid`, `r2_p6_thin_colored_lines`.
Looked at (200% zoom): p2/p6 E3x c100 444 keeps pink/red/green/blue identities (the 75 ppi 4:2:0 failure is gone); p3b pencil equal to reference for E3x; p1 E3x and E5a crisp, E1 150 soft, E5b degraded; p4 E1 150 444 clean, E3x slightly darker/blotchier pen strokes and a mottled highlighter edge, E5 text bolder than reference even after `_t3`.

## Recommendation (priorities: colour identity ≈ 0, tiny-text legibility, natural look, bytes)
**Per-page rule (`pageselect.py`, samples `samples/sample_R2_RECOMMENDED_per_page_selection.pdf`):**
1. blank -> empty page; 2. no-colour page -> E3 mask300 + 100 ppi gray layer (no chroma risk);
3. page with a highlighter region (pale saturated region >= 5000 px after 25 px opening) or ink-heavy art page -> **E1: JPEG 150 ppi q55 4:4:4**;
4. everything else -> **E3x: 300 ppi JBIG2-generic ink mask + 100 ppi q40 4:4:4 colour layer (mean over ink cells, push-pull fill)** (`/Mask` on the colour image).
Mixed result over the 66 pages: **102 KB/page mean (p90 135), 12.6 MB per 124 pages (E0 61 MB), colour-ID changed 0.31%** (0.24% with the 150 ppi layer at 151 KB), highlighter pages natural (hl dE 5.6/4.4), SSIM 0.966, OCR300 0.961, Epson 0.83 pos / 0.867 light (equal to E1 150 within noise; sharper tiny text than E1 150 in the panel). If tiny light text matters more than size: E1 200 q55 4:4:4 (227 KB, Epson 0.843/0.900), the safe single-scheme fallback, or the E3x 150 ppi layer (153 KB).
Rejected after this round: E3 with 4:2:0 or 75 ppi colour layers, dominant-colour cells, dilated mask, E4, E2 < k=32, E5 (no gain), E5b (hurts light text).
Single-scheme alternative if per-page logic is unwanted: E1 150 ppi q55 4:4:4 (152 KB, 4x smaller than E0; colour 0.3%; natural everywhere).

## PDF construction changes in Round 2
Same objects as above with: colour layer `/Fg` JPEG now `subsampling=0` (4:4:4) at `round(W*100/300)` px, q40; E1 base `DeviceRGB` 4:4:4 q55 at 150 ppi. E5: `/Im0` JPEG (4:4:4, 150 ppi) drawn first, then `r g b rg q W 0 0 H 0 0 cm /Mt Do Q` with `/Mt` an `/ImageMask` JBIG2Decode (generic) at 300 ppi. New code: `encoders.py` (`color_layer(mode=)`, `neutral_text_mask`, `enc_hybrid`), configs `E3r_*`, `E3x_*`, `E5*` in `configs.py`.
