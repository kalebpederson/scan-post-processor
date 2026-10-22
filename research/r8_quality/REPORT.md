# R8: `--quality 1..10`

> Figures and per-page rows are not published (private scans). This directory holds the scripts and aggregate numbers only
> (`*_summary.json`); renders and per-page rows are in the git-ignored `work/`.

## Answer

`--quality N` (integer 1..10, default 5) selects how the *cleaned* page is encoded. Level 5 is the previous behaviour, byte for byte
(11 sample PDFs compared with the output of the unmodified code, only the random trailer `/ID` differs, as it does between two runs of
the old code). Cleaning (paper whitening, ink gate, deskew, duplex, folds, blank dropping) is not touched.

| level | what is stored (`auto` encoder) | KB/page | x level 5 |
|---|---|---|---|
| 1 | 300 ppi JBIG2 mask + 75 ppi q25 colour layer (JPEG pages 150 ppi q35) | 54 | 0.65 |
| 2 | mask + 75 ppi q40 (JPEG pages 150 ppi q40) | 63 | 0.76 |
| 3 | mask + 100 ppi q25 (150 ppi q45) | 69 | 0.83 |
| 4 | mask + 100 ppi q32 (150 ppi q50) | 76 | 0.91 |
| **5** | mask + 100 ppi q40 (150 ppi q55) = old output | **83** | **1.00** |
| 6 | mask + 150 ppi q40 (150 ppi q58) | 129 | 1.56 |
| 7 | every page a JPEG, 200 ppi q58 | 217 | 2.61 |
| 8 | JPEG 300 ppi q58 | 393 | 4.74 |
| 9 | JPEG 300 ppi q75 | 486 | 5.87 |
| 10 | JPEG 300 ppi q92 (near-lossless) | 769 | 9.28 |

Bytes and every fidelity measure (SSIM, PSNR, colour dE, ink dE) are monotone over 1..10; steps are about 1.1x below level 5 (there is little
room: the mask alone is ~25 KB and the layer cannot get much coarser without colour errors) and 1.25-1.8x above. All JPEGs are
baseline sequential 4:4:4; masks are JBIG2 generic region (CCITT G4 without `jbig2`); the set of image filters in any output is a subset of
{DCTDecode, JBIG2Decode, CCITTFaxDecode}. Flate lossless (2 MB/page) was measured and rejected: level 10 gets within SSIM 0.0009 at 0.4x
the size. 4:2:0, JPX, JXL, WebP and lossy JBIG2 are not used.

Why the ladder switches from masks to full-page JPEG at level 7: the 1-bit mask loses anti-aliasing and faint ink (20.6 % of the faint ink
pixels, the "faint loss" column; SSIM of any mask page tops out at ~0.98 however fine the colour layer), and a full-page JPEG at 200 ppi
beats a mask page at the same size on SSIM (0.984 vs 0.979 at ~220 KB) and colour dE. The price is edge sharpness (0.82 vs 0.99 at level 6):
JPEG at 200 ppi is softer than a 1-bit 300 ppi mask. Level 8 is back at the scan's resolution and gets both. At 300 ppi even q40 beats
250 ppi q85 on SSIM, PSNR, dE and edge sharpness at 70 % of the size, so 250 ppi has no level.

Wall time: no measurable dependence on the level (3 passes of 11 PDFs, min wall 25.6-29.3 s per pass, level 5: 26.8 s; levels 7-10 about 4 % faster
because no `jbig2` process is spawned; encode stage per page 0.9-1.2 s at levels 1-6 vs 0.6 s at 7-10 on the loaded dev machine).

## Method

* **Pages (58).** The 26 native-300 ppi pages of the private development corpus (`research/corpus`; the 150/200 ppi pages are upsampled
  and would hide resolution effects) and the first 3 pages of each of 11 sample PDFs (scanner output of the same families; plain scans).
  Cleaned with the shipped pipeline (`process_array`, flatten, no resize), no page was blank. The level-5 rule classified them
  as 50 colour (E3x), 5 gray (E3), 3 JPEG pages (E1: highlighter / ink-heavy).
* **Candidates.** `study.py`: sweeps of the knobs the encoders have (colour layer ppi 75-200 x q 25-70; full-page JPEG 150-300 ppi x
  q 25-96; 200 ppi mask; lossless Flate with PNG predictor) in `sweep` and `sweep2`, then the 10 levels as the shipped code encodes them (`ladder`).
* **Fidelity (vs the cleaned full-resolution page).** The encoded page is decoded back to pixels (`recon`: JPEG decoded, layer bilinearly
  upsampled, mask recomputed - it is lossless - and the colour layer shown through it) and compared. A poppler render was tried first and
  rejected: even lossless Flate at 300 ppi came back with SSIM 0.986 / dE 6 because the renderer resamples at 1:1; the decoded-pixel
  comparison gives exactly 1.0 for lossless and is renderer independent. Metrics: SSIM and PSNR on luminance; CIEDE2000 over ink pixels
  (reference min channel < 150, "dE ink") and over coloured ink (also C* > 25, "colour dE"), up to 150k sampled pixels; stroke edge sharpness
  = sum of Sobel gradient at the reference's strong edges, render / reference; solid and faint ink loss (reference ink rendered white),
  speckle (reference-white rendered non-white).
* **Legibility.** tesseract (psm 3, eng) word recall: words of the cleaned page OCR'd at conf >= 90 must reappear in the OCR of a 300 dpi poppler
  render of the encoded PDF (R1/R3 method), pages with >= 20 reference words.
* **End to end.** `run_set.sh` / `run_levels.sh`: the real CLI on the 11 sample PDFs (38 output pages after blank dropping) for each level;
  `cmp_ids.py` for byte identity at level 5; `run_timing.sh` for wall time (3 passes, shared 16-CPU machine).
* Not re-measured here, and relied on: R1 (150 ppi is the floor for light printed text; hence no JPEG page below 150 ppi at any level) and
  R3 (mask below 300 ppi loses light text; 4:2:0 swaps pen colours; 75 ppi colour layer leaves ~1 % of stroke components with changed colour
  and was only used at the two smallest levels).

## Tables

Ladder (the shipped levels), all 58 pages, then per class of the level-5 rule (`ladder_summary.json`). KB = image bytes per page (PDF overhead ~1.5 KB
not included). Edge = stroke edge sharpness, loss = faint-ink loss, OCR = word recall.

| cand | n | KB mean | KB med | KB p90 | SSIM | PSNR | dE ink | dE colour | edge sharp | solid loss | faint loss | speckle | OCR | enc s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 58 | 54.0 | 51.0 | 82.9 | 0.9731 | 28.82 | 8.41 | 8.98 | 0.970 | 0.0000 | 0.206 | 0.0023 | 0.964 | 0.94 |
| L2 | 58 | 63.3 | 61.0 | 95.8 | 0.9736 | 28.91 | 8.01 | 8.57 | 0.980 | 0.0000 | 0.206 | 0.0022 | 0.965 | 0.88 |
| L3 | 58 | 68.6 | 65.9 | 103.8 | 0.9748 | 29.23 | 7.78 | 8.31 | 0.977 | 0.0000 | 0.206 | 0.0022 | 0.966 | 0.98 |
| L4 | 58 | 75.6 | 73.6 | 112.5 | 0.9751 | 29.31 | 7.53 | 8.06 | 0.982 | 0.0000 | 0.206 | 0.0021 | 0.964 | 1.00 |
| L5 | 58 | 82.9 | 81.6 | 122.8 | 0.9753 | 29.37 | 7.36 | 7.86 | 0.985 | 0.0000 | 0.206 | 0.0021 | 0.960 | 1.04 |
| L6 | 58 | 129.2 | 130.3 | 179.3 | 0.9777 | 30.26 | 6.28 | 6.83 | 0.990 | 0.0000 | 0.206 | 0.0021 | 0.965 | 1.22 |
| L7 | 58 | 216.5 | 218.3 | 291.0 | 0.9846 | 33.36 | 6.17 | 4.67 | 0.823 | 0.0000 | 0.000 | 0.0062 | 0.962 | 0.62 |
| L8 | 58 | 393.1 | 395.2 | 526.7 | 0.9955 | 42.21 | 2.73 | 2.38 | 0.993 | 0.0000 | 0.000 | 0.0020 | 0.963 | 0.64 |
| L9 | 58 | 486.3 | 484.8 | 651.7 | 0.9972 | 44.79 | 2.11 | 1.89 | 0.995 | 0.0000 | 0.000 | 0.0009 | 0.973 | 0.58 |
| L10 | 58 | 769.1 | 788.7 | 1042.2 | 0.9991 | 50.74 | 1.33 | 1.14 | 0.997 | 0.0000 | 0.000 | 0.0001 | 0.964 | 0.61 |

class e1 (3 pages):
| cand | n | KB mean | KB med | KB p90 | SSIM | PSNR | dE ink | dE colour | edge sharp | solid loss | faint loss | speckle | OCR | enc s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 3 | 106.6 | 111.2 | 112.8 | 0.9207 | 28.11 | 8.27 | 4.27 | 0.581 | 0.0000 | 0.000 | 0.0435 | 0.970 | 0.46 |
| L2 | 3 | 113.1 | 117.5 | 120.7 | 0.9219 | 28.19 | 8.16 | 4.16 | 0.583 | 0.0000 | 0.000 | 0.0431 | 0.957 | 0.45 |
| L3 | 3 | 120.0 | 123.9 | 129.1 | 0.9232 | 28.22 | 8.05 | 4.06 | 0.586 | 0.0000 | 0.000 | 0.0420 | 0.949 | 0.54 |
| L4 | 3 | 126.7 | 130.5 | 137.0 | 0.9238 | 28.32 | 7.98 | 4.00 | 0.587 | 0.0000 | 0.000 | 0.0409 | 0.970 | 0.46 |
| L5 | 3 | 133.2 | 137.0 | 144.6 | 0.9246 | 28.38 | 7.90 | 3.95 | 0.589 | 0.0000 | 0.000 | 0.0402 | 0.951 | 0.63 |
| L6 | 3 | 138.2 | 141.6 | 150.6 | 0.9251 | 28.42 | 7.86 | 3.91 | 0.590 | 0.0000 | 0.000 | 0.0397 | 0.937 | 0.56 |
| L7 | 3 | 233.7 | 229.4 | 279.5 | 0.9468 | 30.13 | 6.47 | 3.28 | 0.674 | 0.0000 | 0.000 | 0.0336 | 0.957 | 0.58 |
| L8 | 3 | 363.8 | 343.0 | 410.3 | 0.9960 | 43.69 | 2.49 | 1.79 | 0.985 | 0.0000 | 0.000 | 0.0055 | 0.963 | 0.59 |
| L9 | 3 | 460.9 | 428.3 | 524.8 | 0.9974 | 46.29 | 1.88 | 1.32 | 0.990 | 0.0000 | 0.000 | 0.0036 | 0.957 | 0.39 |
| L10 | 3 | 758.7 | 687.0 | 883.7 | 0.9993 | 52.35 | 1.03 | 0.72 | 0.995 | 0.0000 | 0.000 | 0.0005 | 0.957 | 0.65 |

class e3 (5 pages):
| cand | n | KB mean | KB med | KB p90 | SSIM | PSNR | dE ink | dE colour | edge sharp | solid loss | faint loss | speckle | OCR | enc s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 5 | 26.5 | 22.1 | 40.8 | 0.9902 | 31.46 | 8.81 | 20.46 | 0.981 | 0.0000 | 0.225 | 0.0000 | 0.989 | 0.74 |
| L2 | 5 | 30.7 | 26.0 | 46.4 | 0.9904 | 31.53 | 8.44 | 20.58 | 0.991 | 0.0000 | 0.225 | 0.0000 | 0.994 | 0.59 |
| L3 | 5 | 32.4 | 27.2 | 48.9 | 0.9907 | 31.74 | 8.33 | 20.56 | 0.986 | 0.0000 | 0.225 | 0.0000 | 0.993 | 0.64 |
| L4 | 5 | 35.6 | 30.2 | 53.2 | 0.9908 | 31.81 | 8.12 | 20.42 | 0.991 | 0.0000 | 0.225 | 0.0000 | 0.993 | 0.52 |
| L5 | 5 | 38.9 | 33.4 | 57.7 | 0.9909 | 31.87 | 7.94 | 20.17 | 0.996 | 0.0000 | 0.225 | 0.0000 | 0.991 | 0.65 |
| L6 | 5 | 60.3 | 53.5 | 85.7 | 0.9919 | 32.75 | 6.89 | 19.71 | 0.998 | 0.0000 | 0.225 | 0.0000 | 0.998 | 0.88 |
| L7 | 5 | 125.7 | 126.4 | 163.0 | 0.9924 | 35.00 | 7.59 | 9.75 | 0.814 | 0.0000 | 0.000 | 0.0023 | 0.993 | 0.27 |
| L8 | 5 | 242.7 | 243.3 | 319.4 | 0.9974 | 43.54 | 3.12 | 4.81 | 0.992 | 0.0000 | 0.000 | 0.0012 | 0.999 | 0.43 |
| L9 | 5 | 290.5 | 287.2 | 387.4 | 0.9983 | 46.08 | 2.44 | 4.56 | 0.994 | 0.0000 | 0.000 | 0.0006 | 0.997 | 0.22 |
| L10 | 5 | 437.5 | 424.0 | 602.8 | 0.9995 | 52.93 | 1.50 | 2.49 | 0.997 | 0.0000 | 0.000 | 0.0000 | 0.999 | 0.25 |

class e3x (50 pages):
| cand | n | KB mean | KB med | KB p90 | SSIM | PSNR | dE ink | dE colour | edge sharp | solid loss | faint loss | speckle | OCR | enc s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L1 | 50 | 53.6 | 52.4 | 79.3 | 0.9746 | 28.59 | 8.38 | 8.34 | 0.993 | 0.0000 | 0.216 | 0.0000 | 0.961 | 0.99 |
| L2 | 50 | 63.6 | 62.3 | 93.2 | 0.9751 | 28.70 | 7.95 | 7.87 | 1.003 | 0.0000 | 0.216 | 0.0000 | 0.962 | 0.94 |
| L3 | 50 | 69.1 | 67.0 | 101.0 | 0.9763 | 29.03 | 7.71 | 7.58 | 0.999 | 0.0000 | 0.216 | 0.0000 | 0.964 | 1.04 |
| L4 | 50 | 76.6 | 74.2 | 111.0 | 0.9766 | 29.12 | 7.44 | 7.32 | 1.004 | 0.0000 | 0.216 | 0.0000 | 0.961 | 1.08 |
| L5 | 50 | 84.3 | 81.7 | 120.2 | 0.9768 | 29.18 | 7.27 | 7.11 | 1.008 | 0.0000 | 0.216 | 0.0000 | 0.957 | 1.10 |
| L6 | 50 | 135.5 | 132.8 | 186.3 | 0.9794 | 30.12 | 6.13 | 5.97 | 1.013 | 0.0000 | 0.216 | 0.0000 | 0.963 | 1.30 |
| L7 | 50 | 224.6 | 222.6 | 292.4 | 0.9861 | 33.39 | 6.01 | 4.35 | 0.833 | 0.0000 | 0.000 | 0.0050 | 0.959 | 0.66 |
| L8 | 50 | 409.9 | 415.2 | 530.4 | 0.9953 | 41.99 | 2.70 | 2.22 | 0.994 | 0.0000 | 0.000 | 0.0018 | 0.960 | 0.66 |
| L9 | 50 | 507.5 | 518.5 | 662.0 | 0.9971 | 44.57 | 2.09 | 1.71 | 0.995 | 0.0000 | 0.000 | 0.0008 | 0.971 | 0.63 |
| L10 | 50 | 802.9 | 848.5 | 1047.3 | 0.9990 | 50.42 | 1.33 | 1.05 | 0.997 | 0.0000 | 0.000 | 0.0000 | 0.961 | 0.65 |

Candidate sweep (58 pages, auto-routed mask pages and plain-JPEG pages; `mask_cP_qQ` = ink mask + P ppi q Q colour layer, `jpeg_pP_qQ` = full page; highlighter pages
stay JPEG 150/q55 in the `mask_` rows) - `sweep_summary.json`:

| cand | n | KB mean | KB med | KB p90 | SSIM | PSNR | dE ink | dE colour | edge sharp | solid loss | faint loss | speckle | OCR | enc s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mask_c75_q25 | 58 | 55.4 | 51.0 | 82.9 | 0.9733 | 28.83 | 8.39 | 8.96 | 0.971 | 0.0000 | 0.206 | 0.0021 | - | 2.29 |
| mask_c75_q40 | 58 | 64.4 | 61.0 | 95.9 | 0.9738 | 28.92 | 7.99 | 8.56 | 0.980 | 0.0000 | 0.206 | 0.0021 | - | 2.05 |
| mask_c100_q25 | 58 | 69.2 | 65.9 | 104.2 | 0.9748 | 29.23 | 7.77 | 8.30 | 0.977 | 0.0000 | 0.206 | 0.0021 | - | 2.30 |
| mask_m200_c100_q40 | 58 | 72.3 | 71.6 | 104.7 | 0.9797 | 30.60 | 7.76 | 7.05 | 0.855 | 0.0000 | 0.009 | 0.0027 | - | 1.37 |
| mask_c100_q40 | 58 | 82.9 | 81.6 | 122.8 | 0.9753 | 29.37 | 7.36 | 7.86 | 0.985 | 0.0000 | 0.206 | 0.0021 | - | 2.05 |
| mask_c100_q55 | 58 | 95.6 | 95.7 | 137.8 | 0.9756 | 29.45 | 7.11 | 7.61 | 0.990 | 0.0000 | 0.206 | 0.0021 | - | 2.17 |
| jpeg_p150_q25 | 58 | 97.2 | 96.1 | 129.3 | 0.9673 | 30.04 | 9.19 | 7.34 | 0.720 | 0.0000 | 0.000 | 0.0202 | - | 0.32 |
| mask_c125_q40 | 58 | 105.9 | 106.7 | 148.2 | 0.9766 | 29.82 | 6.76 | 7.24 | 0.990 | 0.0000 | 0.206 | 0.0021 | - | 2.49 |
| jpeg_p150_q35 | 58 | 113.3 | 112.6 | 151.9 | 0.9712 | 30.46 | 8.72 | 6.81 | 0.730 | 0.0000 | 0.000 | 0.0164 | - | 0.40 |
| mask_c125_q55 | 58 | 123.7 | 125.2 | 174.2 | 0.9769 | 29.90 | 6.51 | 6.99 | 0.994 | 0.0000 | 0.206 | 0.0021 | - | 2.42 |
| jpeg_p150_q45 | 58 | 126.8 | 126.3 | 170.3 | 0.9740 | 30.64 | 8.43 | 6.44 | 0.736 | 0.0000 | 0.000 | 0.0139 | - | 0.35 |
| mask_c150_q40 | 58 | 128.9 | 130.3 | 179.3 | 0.9777 | 30.25 | 6.29 | 6.83 | 0.990 | 0.0000 | 0.206 | 0.0021 | - | 2.97 |
| jpeg_p150_q55 | 58 | 139.9 | 139.4 | 188.4 | 0.9750 | 30.90 | 8.24 | 6.19 | 0.740 | 0.0000 | 0.000 | 0.0120 | - | 0.43 |
| mask_c150_q55 | 58 | 152.0 | 152.3 | 216.3 | 0.9779 | 30.36 | 6.01 | 6.60 | 0.993 | 0.0000 | 0.206 | 0.0021 | - | 2.57 |
| jpeg_p150_q70 | 58 | 168.2 | 167.7 | 228.5 | 0.9775 | 31.21 | 7.92 | 5.80 | 0.747 | 0.0000 | 0.000 | 0.0090 | - | 0.38 |
| jpeg_p200_q40 | 58 | 181.0 | 181.7 | 240.0 | 0.9823 | 32.94 | 6.53 | 5.18 | 0.818 | 0.0000 | 0.000 | 0.0084 | - | 0.51 |
| mask_c150_q70 | 58 | 187.1 | 191.1 | 262.2 | 0.9782 | 30.46 | 5.72 | 6.34 | 0.995 | 0.0000 | 0.206 | 0.0021 | - | 2.46 |
| jpeg_p200_q55 | 58 | 209.7 | 211.2 | 280.5 | 0.9843 | 33.29 | 6.23 | 4.74 | 0.822 | 0.0000 | 0.000 | 0.0066 | - | 0.45 |
| mask_c200_q55 | 58 | 222.0 | 227.8 | 315.9 | 0.9794 | 31.06 | 5.16 | 5.84 | 1.005 | 0.0000 | 0.206 | 0.0021 | - | 3.65 |
| jpeg_p200_q70 | 58 | 250.8 | 254.2 | 342.5 | 0.9860 | 33.64 | 5.92 | 4.39 | 0.826 | 0.0000 | 0.000 | 0.0050 | - | 0.44 |
| mask_c200_q70 | 58 | 274.9 | 286.0 | 386.9 | 0.9797 | 31.17 | 4.87 | 5.58 | 1.006 | 0.0000 | 0.206 | 0.0021 | - | 3.84 |
| jpeg_p200_q80 | 58 | 298.6 | 305.1 | 414.2 | 0.9873 | 33.90 | 5.71 | 4.08 | 0.829 | 0.0000 | 0.000 | 0.0040 | - | 0.46 |
| jpeg_p250_q60 | 58 | 305.2 | 310.5 | 414.7 | 0.9896 | 35.32 | 4.90 | 3.74 | 0.872 | 0.0000 | 0.000 | 0.0038 | - | 0.54 |
| jpeg_p250_q75 | 58 | 371.3 | 377.1 | 509.4 | 0.9909 | 35.72 | 4.59 | 3.39 | 0.874 | 0.0000 | 0.000 | 0.0030 | - | 0.51 |
| jpeg_p300_q60 | 58 | 400.1 | 401.0 | 537.6 | 0.9957 | 42.41 | 2.67 | 2.32 | 0.993 | 0.0000 | 0.000 | 0.0019 | - | 0.45 |
| jpeg_p250_q85 | 58 | 464.2 | 474.6 | 642.4 | 0.9919 | 36.03 | 4.35 | 3.11 | 0.876 | 0.0000 | 0.000 | 0.0025 | - | 0.64 |
| jpeg_p300_q75 | 58 | 486.3 | 484.8 | 651.7 | 0.9972 | 44.79 | 2.11 | 1.89 | 0.995 | 0.0000 | 0.000 | 0.0009 | - | 0.50 |
| jpeg_p300_q85 | 58 | 601.1 | 612.0 | 807.7 | 0.9983 | 47.43 | 1.69 | 1.48 | 0.996 | 0.0000 | 0.000 | 0.0003 | - | 0.48 |
| jpeg_p300_q90 | 58 | 711.5 | 731.8 | 961.1 | 0.9989 | 49.55 | 1.43 | 1.27 | 0.997 | 0.0000 | 0.000 | 0.0001 | - | 0.49 |
| jpeg_p300_q93 | 58 | 816.5 | 833.4 | 1104.3 | 0.9992 | 51.51 | 1.26 | 1.06 | 0.998 | 0.0000 | 0.000 | 0.0000 | - | 0.50 |
| jpeg_p300_q96 | 58 | 1049.8 | 1064.6 | 1425.3 | 0.9996 | 55.06 | 1.01 | 0.77 | 0.998 | 0.0000 | 0.000 | 0.0000 | - | 0.51 |
| flate_300 | 58 | 1928.9 | 1920.0 | 2874.8 | 1.0000 | 138.13 | 0.00 | 0.00 | 1.000 | 0.0000 | 0.000 | 0.0000 | - | 2.32 |

Second sweep, the exact settings the ladder uses and neighbours - `sweep2_summary.json`:

| cand | n | KB mean | KB med | KB p90 | SSIM | PSNR | dE ink | dE colour | edge sharp | solid loss | faint loss | speckle | OCR | enc s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mask_c75_q32 | 58 | 59.8 | 55.8 | 89.4 | 0.9736 | 28.88 | 8.14 | 8.74 | 0.976 | 0.0000 | 0.206 | 0.0021 | - | 1.44 |
| mask_c100_q32 | 58 | 76.0 | 73.6 | 113.9 | 0.9751 | 29.31 | 7.52 | 8.06 | 0.982 | 0.0000 | 0.206 | 0.0021 | - | 1.36 |
| mask_c125_q48 | 58 | 115.5 | 116.4 | 161.3 | 0.9768 | 29.87 | 6.61 | 7.11 | 0.992 | 0.0000 | 0.206 | 0.0021 | - | 1.77 |
| jpeg_p150_q40 | 58 | 120.0 | 119.2 | 161.0 | 0.9723 | 30.59 | 8.58 | 6.61 | 0.733 | 0.0000 | 0.000 | 0.0151 | - | 0.27 |
| jpeg_p150_q50 | 58 | 133.5 | 133.1 | 179.7 | 0.9743 | 30.81 | 8.33 | 6.32 | 0.738 | 0.0000 | 0.000 | 0.0129 | - | 0.29 |
| jpeg_p150_q62 | 58 | 150.8 | 149.8 | 203.8 | 0.9762 | 31.04 | 8.11 | 6.03 | 0.743 | 0.0000 | 0.000 | 0.0107 | - | 0.26 |
| jpeg_p200_q65 | 58 | 233.7 | 236.3 | 316.6 | 0.9854 | 33.52 | 6.05 | 4.52 | 0.825 | 0.0000 | 0.000 | 0.0055 | - | 0.38 |
| jpeg_p250_q65 | 58 | 323.0 | 328.6 | 440.3 | 0.9900 | 35.44 | 4.82 | 3.60 | 0.873 | 0.0000 | 0.000 | 0.0036 | - | 0.46 |
| jpeg_p300_q40 | 58 | 331.8 | 341.9 | 438.4 | 0.9937 | 40.35 | 3.26 | 2.97 | 0.992 | 0.0000 | 0.000 | 0.0033 | - | 0.35 |
| jpeg_p250_q72 | 58 | 355.2 | 361.0 | 486.3 | 0.9907 | 35.64 | 4.64 | 3.45 | 0.874 | 0.0000 | 0.000 | 0.0031 | - | 0.50 |
| jpeg_p300_q50 | 58 | 365.2 | 372.4 | 485.3 | 0.9948 | 41.40 | 2.94 | 2.60 | 0.992 | 0.0000 | 0.000 | 0.0025 | - | 0.37 |
| jpeg_p300_q82 | 58 | 558.7 | 564.7 | 749.2 | 0.9980 | 46.60 | 1.82 | 1.62 | 0.996 | 0.0000 | 0.000 | 0.0005 | - | 0.36 |
| jpeg_p300_q92 | 58 | 769.1 | 788.7 | 1042.2 | 0.9991 | 50.74 | 1.33 | 1.14 | 0.997 | 0.0000 | 0.000 | 0.0001 | - | 0.36 |

End to end on 11 sample PDFs (38 pages), output size relative to level 5 (`levels_summary.json`; per-PDF range in the last columns):

| level | total KB | KB/page | x level 5 | min..max per-PDF x L5 | encode CPU s | total CPU s | wall s |
|---|---|---|---|---|---|---|---|
| 1 | 1879 | 49.5 | 0.63 | 0.59..0.67 | 26.4 | 168.2 | 165.2 |
| 2 | 2236 | 58.8 | 0.75 | 0.73..0.78 | 28.3 | 184.8 | 186.3 |
| 3 | 2424 | 63.8 | 0.82 | 0.80..0.83 | 27.4 | 164.3 | 152.7 |
| 4 | 2693 | 70.9 | 0.91 | 0.90..0.91 | 29.0 | 172.6 | 170.8 |
| 5 | 2970 | 78.2 | 1.00 | 1.00..1.00 | 24.2 | 142.7 | 125.7 |
| 5b | 2970 | 78.2 | 1.00 | 1.00..1.00 | 13.5 | 84.3 | 51.8 |
| 6 | 4826 | 127.0 | 1.62 | 1.59..1.72 | 37.0 | 184.9 | 179.3 |
| 7 | 8426 | 221.7 | 2.84 | 2.57..3.73 | 17.2 | 146.5 | 107.2 |
| 8 | 14864 | 391.1 | 5.00 | 3.11..7.22 | 12.2 | 103.0 | 50.1 |
| 9 | 18465 | 485.9 | 6.22 | 3.60..8.51 | 8.1 | 69.7 | 31.2 |
| 10 | 29270 | 770.3 | 9.85 | 5.21..12.58 | 8.4 | 69.5 | 45.8 |

Wall time (`timing_summary.json`, seconds for the 11 PDFs, three passes; the machine was shared so single passes vary +-15 %):

| level | wall s per pass (11 PDFs) | min | median | min / level-5 min |
|---|---|---|---|---|
| 1 | 26.9, 33.2, 26.5 | 26.5 | 26.9 | 0.99 |
| 2 | 29.1, 30.3, 26.5 | 26.5 | 29.1 | 0.99 |
| 3 | 29.3, 30.3, 29.6 | 29.3 | 29.6 | 1.09 |
| 4 | 27.1, 33.5, 26.9 | 26.9 | 27.1 | 1.00 |
| 5 | 27.3, 29.4, 26.8 | 26.8 | 27.3 | 1.00 |
| 6 | 30.4, 33.7, 27.6 | 27.6 | 30.4 | 1.03 |
| 7 | 29.5, 30.9, 25.6 | 25.6 | 29.5 | 0.96 |
| 8 | 30.4, 27.4, 25.7 | 25.7 | 27.4 | 0.96 |
| 9 | 30.7, 25.7, 25.6 | 25.6 | 25.7 | 0.96 |
| 10 | 31.5, 25.6, 25.6 | 25.6 | 25.6 | 0.96 |

## Limitations

* Fidelity is measured against the *cleaned* page (as the request asks); how good the cleaning is, is R2. `--quality 10` cannot restore what the ink
  gate removed. The cleaned pages come from 300 ppi scanner JPEGs, so the 300 ppi levels partly re-encode scanner JPEG noise.
* SSIM jumps between levels 7 and 8 mostly because the 300 ppi page is on the reference's pixel grid (no resampling); 250 ppi would sit between
  by metric but is dominated by 300 ppi q40 on bytes, so it was dropped. Metrics are on-grid pixel comparisons, not a viewer simulation; the
  bilinear upsample of low-resolution layers is a neutral choice and viewers differ (R3: renderers agree within 2-3/255 mean).
* OCR recall does not separate levels (all 0.960-0.973; noise ~+-0.02) because the reference words are the cleaned page's own and tesseract is
  insensitive at 300 ppi renders; legibility at level 1 rests on R1/R3 (mask stays 300 ppi, JPEG pages >= 150 ppi), not on this table. Tiny light
  halftone text (R1's hardest class, Epson 600 dpi scans) was not re-run.
* Level 1-2 use the 75 ppi colour layer that R3 round 2 found too coarse for "colour identity ~ 0" (1.0 % of stroke components changed, 0.46 % swaps
  with 4:4:4); that is the price of the smallest levels, level 3 upward is at 100 ppi. Colour identity itself (component hue test of R3) was not re-measured;
  colour dE is the proxy.
* 58 pages from one family of material (maths worksheets, one school); one scanner (Brother ADS-2200) and a few flatbed pages. Gray (5) and JPEG-route (3) classes are small.
* Sizes are image bytes; the PDF container adds ~1.5 KB per page; with `--ocr` the text layer adds on top.
* Wall times on a shared machine (load average 100+ from other jobs during the runs) have +-15 % noise; the conclusion is only "no dependence".
* Viewers: JPEG 4:4:4 baseline and JBIG2 generic are as portable as before; level 7-10 outputs contain no JBIG2 at all. Nothing was run in Acrobat/Preview.
* Image-file input (JPEG/PNG/WebP) does not use the page encoders, so `--quality` does not apply to it.
