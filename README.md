# scanpp - scan post-processor

scanpp is a post-processor for scanned paper documents. It was written to run as the post-processing script of GNOME Document
Scanner (simple-scan) on Linux, and also works as a command line tool. It turns large colour scans (300 dpi, 20-40 MB for a
multi-page document) into small, clean, readable PDFs: paper white, ink dark, coloured ink kept, blank pages and scanner
edge junk removed. On the development corpus, 885.6 MB of scan PDFs became 122.3-123.9 MB (median 67 KB per page).

## Goals

The goals, as set by the author:

* High resolution where it matters (printed text and handwriting stay legible) without large files.
* A legible output resolution chosen from measurements, not by guess.
* Fast: uses all CPU cores, and runs in the background so simple-scan is not blocked.
* White paper and black ink.
* Bleed-through from the other side of the sheet kept to a minimum.
* Colour coding preserved: coloured pens, highlighter and pencil keep their colour and identity.
* Output that opens in any PDF viewer without extra codecs: baseline JPEG and lossless JBIG2 generic region / CCITT G4
  only. No JPEG 2000, JPEG XL, WebP or lossy JBIG2.
* The original is always backed up before anything is changed.
* Every default is decided by a measured study and covered by unit and integration tests.

Configurable: paper colour (flatten to white, the default, or keep the tint), OCR (off by default), blank-page dropping
(on, with a threshold), deskew, duplex bleed-through suppression, fold-crease removal.

Per page: decode embedded image -> paint over scanner edge strips / spiral coil -> remove fold creases -> blank-page detection (drop) ->
skew estimate -> flat-field cleaning (paper to white, ink kept, hue preserved) -> deskew -> encode the full-resolution
cleaned page (the encoder does its own resampling, see "Encoders"). Page size (MediaBox, /Rotate) is preserved.
PDFs that are not plain scans (any text/fonts, or not exactly one image per page) are logged and left untouched.

## What it is for, and what it is not for

For:

* Multi-page scans of paper documents: printed handouts, notes, worksheets, letters, with print, pen, pencil and highlighter, on white
  or coloured paper.
* Scanned in colour at 300 dpi (200-300 works).
* Linux with simple-scan (and `notify-send` for notifications).

Not for:

* Photographs or artwork. The pipeline whitens the paper and thresholds ink, which destroys a photo. Photo-like image input
  comes out tonally altered (acceptance report, item 7).
* A general image post-processor. Single JPEG/PNG/WebP files are accepted and cleaned with the same document pipeline as a
  convenience, but the target is scanned-document PDFs and image-file handling is much less tested.
* Born-digital PDFs, or PDFs that already have text or fonts. They are detected and left untouched.
* Replacing a full OCR suite. OCR is an optional searchable layer; handwriting is not recognised usefully.
* Other platforms. Only Linux is tested. The core CLI is plain Python and may work elsewhere, which is untested.

## What was tested

**Development corpus.** 1,969 pages in 291 scanned PDFs of secondary-school maths material (teacher notes and handouts,
worksheets, student formatives, tests) collected over several school years. Content: print, coloured pens, pencil, highlighter;
white, pink, yellow, blue and green paper; spiral-bound notebooks; mostly double-sided. The pages are US Letter (the surveyed
files). Scanners: a Brother ADS-2200 sheet-fed duplex scanner (300 dpi colour, via simple-scan) and an Epson flatbed (used for the 600 and
1200 dpi reference scans of the resolution study). The acceptance run used 309 input files (1,917 pages in the 275 files that were processed, 34 files skipped as not plain scans).

This corpus is private (it contains student work and names) and is not published. The reports publish only aggregate numbers;
the figures and page crops they mention are not in the repository.

| study | question | key result |
|---|---|---|
| [R1 resolution](research/r1_resolution/REPORT.md) | Output ppi vs legibility; does scanner JPEG quality matter? | 200 ppi default (150 is the floor). Light-halftone small text word recall 0.87 @150, 0.91 @200, 0.92 @250, 0.91 @300; handwriting fine at 150; 388 reference words |
| [R2 cleaning](research/r2_cleaning/REPORT.md) | Paper white, ink black, faint content and colour kept, bleed-through gone | Paper whiteness 1.000 (also on pink, blue-grey, yellow paper); faint-content retention 0.85 vs 0.66 for the old ImageMagick levels; bleed-only survival 0.016 vs 0.098 |
| [R3 encoding](research/r3_encoding/REPORT.md) | How to store a cleaned page | Per page: empty / JBIG2 mask + gray layer / JPEG / JBIG2 mask + colour layer. 102 KB per page mean in the study (12.6 MB per 124 pages vs 61 MB for the old command); colour identity changed 0.31 %; checked in poppler, MuPDF, Ghostscript, PDFium, pdf.js |
| [R4 geometry](research/r4_geometry/REPORT.md) | Deskew, blank pages, scanner edge artefacts on 1,969 pages | Skew reliable on 92.5 % of pages (p50 0.10 deg, max 3.04 deg); 247 blank pages (12.5 %), 0 false, 0 missed; spiral coil on 34.5 % of pages |
| [R5 duplex](research/r5_duplex/REPORT.md) | Use the other side of a double-sided sheet to remove marker bleed-through | 0 false pair accepts out of 43,685 non-pairs; 138 adjacent pairs accepted corpus-wide; ~0.24 s CPU per sheet |
| [R6 fold creases](research/r6_fold/REPORT.md) | Remove gray fold creases without touching rules, grids or text | 9/9 relevant creases on a folded test document, 0 false; 0 false detections on 1,946 of 1,951 evaluated corpus pages (the other 5 fire: 3 genuine creases on pink paper, 2 booklet edges) |
| [R7 flatbed edges](research/r7_flatbed/REPORT.md) | Edge artefacts of flatbed scans of spiral notebooks (sheet-edge line above the lid, coil-end shadow, frame hairline, corner dots) | Four conservative paint rules; 0 content erased in the reviewed pages; pages without a firing rule are byte-identical to before |
| [brother_live](research/brother_live/REPORT.md) | The pipeline on a live 22-page Brother ADS-2200 duplex scan | 4 blank pages dropped correctly, skew up to 3.08 deg corrected, small-print words lost 2.2-3.0 % (2.2 % by the scanner JPEG alone). Open issue: saturated highlighter / marker bleed is kept; top-edge blotches on coloured paper |
| [acceptance](research/acceptance/REPORT.md) | Whole-corpus run, three rounds | See below |

**Whole-corpus outcome** (acceptance report, final run with duplex): 309 files, all exit 0, 275 processed, 34 skipped
(not plain scans); 885.6 MB -> 122.3 MB (x0.138); median 66.2 KB per kept page (p90 98.5 KB); 247 blank pages dropped, all
reviewed, none wrongly; 127 duplex sheets paired, 190 pages cleaned of marker ghosts; a second run on the outputs skips all 309
files byte-identically. Renderers (poppler, MuPDF, Ghostscript, PDFium) agree visually; thin 1-bit-mask strokes differ in
anti-aliasing between them.

**Speed** (16-CPU development machine, `--jobs 16`): a 124-page, 38.6 MB scan takes about 15.5 s with duplex and folds
off; duplex adds 16.6 % (18.1 s) and fold detection a further ~0.1 s CPU per page, so roughly 16-22 s depending on options and machine load. The
simple-scan call itself returns in about 0.2-0.3 s because the work runs in the background.

**Test suite.** 295 tests: unit tests on synthetic images for each stage (cleaning, geometry, blank score, duplex, folds),
encoder and PDF validity tests, integration tests of the CLI, background mode and the simple-scan protocol, safety and backup
tests, truncated-image robustness, OCR, and equivalence of the shipped code with the research prototypes. All 295 pass on the
development machine. Tests that need the private corpus skip themselves: on a fresh clone 275 pass and 20 skip.

**Not tested / known gaps:**

* Other scanners and drivers (the Brother's brightness/contrast settings are not applied by its driver, so their effect is unknown).
* Paper sizes other than US Letter.
* Flatbed scans beyond one user's spiral-notebook scans (an Epson flatbed was used only for reference scans).
* OCR for non-Latin scripts; OCR was tested with tesseract 5.5.0 and English.
* Photos, and very low-resolution scans (below 150 dpi).
* Vertical fold creases and creases at other heights than 1/3, 1/2, 2/3 of the page.
* Non-Linux platforms.

## Install

Developed on Ubuntu 26.04 with Python 3.14; requires Python 3.11 or newer.

    git clone https://github.com/<user>/scan-post-processor
    cd scan-post-processor
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/pip install -e .

`requirements.txt` pins the versions used for development and also holds research-script dependencies (scipy, scikit-image).
The runtime needs only numpy, opencv-python-headless, pillow and pikepdf (see `pyproject.toml`).

System packages (Debian/Ubuntu names):

    sudo apt install simple-scan jbig2enc-tools tesseract-ocr tesseract-ocr-eng libnotify-bin poppler-utils mupdf-tools

* `jbig2enc-tools` (provides `jbig2`): smallest files. Without it, masks fall back to CCITT G4 (about 25 % larger masks).
* `tesseract-ocr` plus `tesseract-ocr-<lang>`: only for `--ocr`.
* `libnotify-bin` (`notify-send`): desktop notifications from simple-scan runs.
* `poppler-utils`, `mupdf-tools`: only used by some tests.

`bin/scan-postprocess` is the wrapper simple-scan calls. It resolves its own real path (so symlinking it elsewhere works) and picks the
interpreter: `$SCANPP_PYTHON` if set, else the repository's `.venv/bin/python`, else `python3`; then it runs `python -m scanpp "$@"`.

## simple-scan setup

simple-scan runs `script MIME_TYPE KEEP_ORIGINAL FILE_PATH [ARGS]` synchronously after saving (the UI waits for it).
Set the preferences (gsettings schema `org.gnome.SimpleScan`):

    gsettings set org.gnome.SimpleScan postproc-enabled true
    gsettings set org.gnome.SimpleScan postproc-script /path/to/scan-post-processor/bin/scan-postprocess
    gsettings set org.gnome.SimpleScan postproc-arguments '--paper keep --keep-blank'   # optional, one string

`postproc-arguments` is passed as ONE argument and shlex-split by scanpp. A non-zero exit is only logged by simple-scan.
Logs: `~/.cache/scanpp/scanpp.log` (rotating) and stderr.

Recommended scan settings: colour, 300 dpi (200-300 works; below 150 dpi is untested). Everything was developed and measured on
300 dpi colour scans. Leave the scanner's own brightness/contrast alone.

## Usage and options

    scanpp [options] INPUT [INPUT ...] [-o OUTPUT]      # normal mode (in place unless -o; -o needs a single input)
    scanpp MIME_TYPE true|false FILE_PATH ["ARGS"]       # simple-scan protocol (auto-detected)

| option | meaning |
|---|---|
| `--paper flatten\|keep` | paper to pure white (default) or keep a uniform paper tint |
| `--drop-blank` / `--keep-blank` | drop blank pages (default) / keep them. If ALL pages are blank, all are kept |
| `--duplex auto\|off` | `auto` (default): in multi-page PDFs use the other side of a double-sided sheet to remove coloured-marker bleed-through - see "Duplex". `off`: every page on its own |
| `--folds auto\|off` | `auto` (default): remove the gray crease line a folded sheet leaves - see "Fold creases". `off`: leave creases alone |
| `--no-deskew`, `--no-edge-clean` | disable straightening / edge-strip + coil painting |
| `--dpi N` | output ppi for the `jpeg`/`jpeg200` encoders ONLY (default 150 / 200; never upsamples). `auto` and `mrc` ignore it (300 ppi mask, 100 ppi colour layer, 150 ppi JPEG pages). Image files (JPEG/PNG/WebP) are still resampled to `--dpi` (default 200) |
| `--jobs N` | worker processes (default: CPU count) |
| `--encoder NAME` | `auto` (default), `mrc`, `jpeg`, `jpeg200` - see "Encoders"; new encoders register in `scanpp/encode.py` |
| `--backup-dir DIR` / `--no-backup` | backup location (default `.originals/` next to the file) / no backup |
| `--ocr` | add an invisible, searchable text layer to PDF pages (tesseract; off by default) - see "OCR" |
| `--ocr-lang LANG` | tesseract language(s) for `--ocr`, e.g. `eng` (default) or `eng+spa` |
| `--ocr-min-conf N` | drop OCR words below tesseract confidence N (0-100, default 60) |
| `--background` / `--foreground` | return at once and process in a detached worker / process before returning (see below) |
| `--no-notify` | no desktop notifications |
| `-v` | verbose, plus total time and per-stage CPU breakdown |
| `--report FILE.json` | per-page stats: skew angle/confidence/applied, blank score/dropped, edge-painted fraction, bytes, stage CPU, and with `--ocr` per page `ocr`: words kept/raw, mean confidence, tesseract CPU and wall seconds |

Single images (JPEG/PNG/WebP) are processed in place in the same format (JPEG q85, PNG optimized lossless
(palette when <=256 colours), WebP lossless) with the same pipeline minus blank dropping. Missing dpi metadata is assumed to be 300.

## Blank pages

Pages whose `blank_score` is below a threshold (default **15**) are dropped (typically blank duplex backs). The score is the
"sharp ink" mass of the page at 100 ppi after edge/coil cleanup: roughly, how many pixels of real pencil/pen/print it carries.
If *every* page of a file is blank, all pages are kept. Each dropped page is logged at INFO with its score and the threshold
(`~/.cache/scanpp/scanpp.log`), and `--report` has `blank_score` for every page, so you can tune from real data.

| what is on the page | typical score | dropped at 15? |
|---|---|---|
| blank paper: dust, bleed-through, shadows | 0-10 (corpus: 247 labelled blanks, median 0.2, p99 22, max 25) | yes (5 of 247 dusty edges score 15-25) |
| 0.1 in pencil mark | 5-15 | mostly |
| 0.5 in dark / light pencil line | 25-50 / 2-30 | kept / mostly dropped |
| 1-2 in dark pencil line | 50-135 (live scan: up to 160) | kept |
| 2 in light pencil line | 10-40 (live scan: 65-75) | mostly kept |
| lone page number "7" / "12", printed | 15-45 / 35-70 | kept (92% / 100%) |
| one handwritten word, light pencil | 25-110 (median 45) | kept |
| one handwritten word, dark pencil / pen | 125-490 / 215-835 | kept |
| "Page 2 of 2" | 140-180 | kept |

(Synthetic marks painted onto real blank scans, +-30%; live-scan figures in `research/brother_live/REPORT.md`.
A tiny, very light word, ~0.15 in tall in pencil that is barely visible, scores ~5-40 and may be dropped.)

Tune it with `--blank-threshold N` (pages scoring below N are dropped; lower keeps more, 0 keeps everything) or turn dropping off with
`--keep-blank`. In simple-scan's single-string arguments:

    gsettings set org.gnome.SimpleScan postproc-arguments '--blank-threshold 25'     # also drop a lone page number / short line
    gsettings set org.gnome.SimpleScan postproc-arguments '--blank-threshold 8'      # keep nearly everything with any mark
    gsettings set org.gnome.SimpleScan postproc-arguments '--keep-blank'             # never drop a page

## Duplex (`--duplex auto|off`)

A coloured marker on one side of a sheet shows through, paler and mirrored, on the other side, and a single page cannot tell that ghost
from real ink. For multi-page PDFs scanpp therefore looks at every pair of consecutive pages and checks whether they are the two sides of
one sheet (the colour ink of one side must line up, mirrored, with a faint colour layer on the other side, uniquely and from both
directions or very strongly from one). Only when it is confident, the marker of the front is used to erase its show-through from the back (and
vice versa) after the normal cleaning. Ink that is darker, more saturated or of another colour than the marker is kept, as is any stroke that
crosses a marker band. No confident pair, a one-page PDF, an image file or any error: the page is processed exactly as without the feature
(`--duplex auto` and `--duplex off` give identical pages for such documents). A page dropped as blank can still be the source of a pair.

Costs about 0.1 s of CPU per page for the analysis plus about 0.4 s per page of an accepted sheet (about 17 % more wall time on a 124-page file with
16 jobs, 18.1 s vs 15.5 s, see `research/r5_duplex/REPORT.md`). The report has `duplex_partner`, `duplex_ncc` and `duplex_removed_px` per page; the log has one
line per accepted sheet. A mask that would remove more than 25 % of a page's ink is discarded.

Limits: marker-grade colour only (thin pens, black print and pencil ghosts are left to the single-page cleaner); faint pen strokes of
the same colour exactly under a bleed band may be lightened; two strong marker notes at the same mirrored spot are not separated.

## Fold creases (`--folds auto|off`)

Paper that was folded for a letter or an envelope shows a thin gray line where the fold was, and the cleaner would turn that faint line
into a hard dark one. scanpp therefore looks for a crease on every page before it is cleaned and, if it finds one, replaces just the crease
pixels with the paper beside them. Ink that crosses the crease is kept. It only acts on a faint line that runs across the full page width at
about 1/3, 1/2 or 2/3 of the page height (a second crease of a tri-fold, and a crease that shows on both sides of a colour sheet, are
accepted a little more easily). Ruled paper, table rules, underlines, graph paper and dashed lines are not touched, and a page without a
detected crease comes out byte-for-byte as with `--folds off`. A page that is only a crease is still dropped as blank.

Costs about 0.05-0.1 s of CPU per page (about 0.03 s more on a page with a crease). The report has `folds` (row as a fraction of the page
height, angle, tier, score) and `fold_px_removed` per page; the log has one line per page with creases removed; `-v` shows the
`fold_detect` / `fold_remove` stage times. Any error in this step is logged and the page is processed as with `--folds off`.
Details and evaluation: `research/r6_fold/REPORT.md`.

Limits: a perfectly straight, uniform printed gray line across the whole page exactly at one of those heights would also be removed; thin
horizontal strokes lying right on a crease (a hairline rule, a fraction bar) may be lightened; creases through dense text are often not found
and are then left as the cleaner makes them; light residue may remain where a crease runs through a line of text; vertical folds and
creases at other positions are not handled. `--folds off` disables the step.

## Encoders

The cleaned page is white paper with ink, so it is stored as an ink mask plus a low-res colour layer where that is
better, and as a plain JPEG where it is not (design and measurements: `research/r3_encoding/REPORT.md`, Round 2).

| `--encoder` | output |
|---|---|
| `auto` (default) | per page: **no colour** -> E3 (mask + 100 ppi gray layer); **highlighter** (pale saturated region >= 5000 px @300 ppi after a 25 px opening) or **ink-heavy art** (ink > 15% / non-white > 30%) -> E1; otherwise **E3x** |
| `mrc` | always E3x (E3 with a gray layer on no-colour pages) |
| `jpeg` | always E1: baseline JPEG, 150 ppi, q55, 4:4:4 (gray JPEG if the page has no colour) |
| `jpeg200` | E1 at 200 ppi ("max legibility") |

E3x per page: a 1-bit ink mask at native resolution, at most 300 ppi (adaptive threshold on darkness d = 255 - min(R,G,B):
ink if `d >= clip(0.35 * local_max_7x7(d), 20, 88)`), JBIG2 **generic-region** coded (`jbig2` from jbig2enc, never symbol
mode), used as an explicit `/Mask` on a 100 ppi q40 4:4:4 baseline JPEG colour layer (ink-only normalized convolution, empty
cells push-pull filled). Chroma is never subsampled: 4:2:0 desaturates thin pen strokes and swaps adjacent colours.
Pages below 300 dpi are never upsampled (mask at native resolution, colour layer at min(100, native) ppi); above 300 dpi the
page is area-downsampled to 300 ppi first. If `jbig2` is missing or fails, the mask is CCITT G4 instead (a bit-identical
render, ~25% larger masks; logged once). Blank pages that are kept get an empty content stream (pure white page).
The per-page report records `scheme` (`e3`, `e3x`, `e1`, `blank`), mask codec and bytes. Output of `auto`/`mrc` is not
re-processed by a second run (images with a `/Mask` are not "plain scans"). Needs `jbig2` (Debian/Ubuntu: `jbig2enc-tools`)
for the smallest files; pdf viewers all render JBIG2 generic regions.

## OCR (`--ocr`, optional, off by default)

Adds an **invisible text layer** so the PDF is searchable and its text can be selected/copied. The page image is not
touched: with and without `--ocr` the renders are pixel-identical (checked with pdftoppm and mutool).

* Per page, in the worker that already holds the page: tesseract runs on the **cleaned, deskewed, upright page at full
  resolution** (gray, before encoding) with `OMP_THREAD_LIMIT=1`, so parallelism stays at page level. One pass yields
  tesseract's text-only PDF (`textonly_pdf=1`, text render mode 3, glyphless font) and a TSV with word confidences.
* Words below `--ocr-min-conf` (default 60) are removed from that text (the other words keep their positions). The rest
  becomes a Form XObject drawn after the image with the matrix that maps tesseract's page onto the real page
  (scaled to the MediaBox, `/Rotate` undone, MediaBox origin applied). The glyphless font is shared by all pages.
* Not OCR'd: dropped blank pages, kept blank pages, image files (JPEG/PNG/WebP cannot carry text). A page where OCR
  finds nothing gets no layer.
* Failure is never fatal: tesseract not installed, a missing language (`--ocr-lang`), a crash or a timeout (300 s) gives
  a warning in the log and the page/file is written without text. Missing tesseract/language is reported once per file.
* Output with a text layer is recognised as scanpp output (the Form XObject carries a `/ScanppOCR` marker), so it is not
  processed again. Digital PDFs and PDFs OCR'd by other tools (fonts present) are still left untouched.
* Languages: only those installed (`tesseract --list-langs`; Debian: `tesseract-ocr-<lang>`). Tested with tesseract 5.5.0.

**Cost** (measured on the development machine (16 CPUs), 16 jobs, tesseract 5.5.0; the machine was shared, so wall times are noisy):
about **1 s of CPU per page** (0.7-1.0 s; 0.7 s at 300 ppi, tesseract is single-threaded per page) and **2-3 KB per
page** (+3% of the output size, e.g. 6440 -> 6628 KB for a 93-page file). A 124-page scan took ~16 s without and
~20 s with OCR. In background mode (the simple-scan default) this runs detached, so simple-scan is not blocked.

**Enable in simple-scan** (arguments are one string, combine with other options):

    gsettings set org.gnome.SimpleScan postproc-arguments '--ocr'
    gsettings set org.gnome.SimpleScan postproc-arguments '--ocr --ocr-lang eng+spa --paper keep'

**Junk control.** tesseract returns garbage for handwriting. On a nearly pure handwriting page it found 89 "words" at mean
confidence 38; at the default threshold 22 remain, mostly math symbols. Across 991 scanned pages of the development corpus the share of
dictionary words kept is 95.1% / 94.2% / 93.5% / 92.5% at thresholds 50 / 60 / 70 / 80, while non-dictionary
alphabetic tokens kept are 46% / 39% / 33% / 25%. 60 was chosen as the point where search recall for printed text is
still ~94% (printed words next to rules/boxes can score 50-70) and most junk is gone; raise it for fewer false hits.

## Background mode (simple-scan)

simple-scan blocks its UI until the post-processor exits (and until its stdout/stderr pipes close), so when invoked
through the simple-scan protocol scanpp runs in the **background** by default:

1. Foreground part (well under a second for a 124-page file): parse args, check the file, cheaply classify it
   (scan PDF or supported image; anything else is left untouched, exit 0), copy it to `.originals/` and verify the
   copy, record a fingerprint (size, mtime, sha256 of the first and last 1 MB), then start a detached worker
   (new session, stdin/stdout/stderr = /dev/null, no inherited fds) and exit 0.
2. The worker takes an exclusive lock (`~/.cache/scanpp/worker.lock`), so consecutive scans are processed one at a
   time (each uses all CPUs), at nice 10. It processes the **backup copy**, validates the result, and before the
   atomic replace re-checks the fingerprint of the live file. If you re-saved the file meanwhile, the result is written
   next to it as `<name>.scanpp.pdf` instead (and you are notified); if you deleted it, the result is discarded.
3. Everything is logged with a job id to `~/.cache/scanpp/scanpp.log` (timings, outcome). `--report` also works.

**Email exception:** simple-scan's Email action saves into a temp dir named `simple-scan-XXXXXX` and attaches the
file as soon as the script returns, so files whose parent directory matches `^simple-scan-[A-Za-z0-9]{6}$` are always
processed in the foreground. `-o` and multiple inputs are also always foreground. The normal CLI is foreground
unless you pass `--background`.

To force foreground for everything, put it in the arguments setting:
`gsettings set org.gnome.SimpleScan postproc-arguments '--foreground'`.

**Notifications** (`notify-send`, app name "scanpp"; only for simple-scan runs, never for the plain CLI): a short
message on success ("file: N pages (M blank dropped), X MB -> Y MB"); a persistent one on failure or when the target
changed/was deleted, with the log path. Foreground (email) runs notify on failure only. Disable with `--no-notify`;
a missing/failing `notify-send` is ignored (and logged). Tests never notify (`SCANPP_NOTIFY=0`, pytest guard).

## Backups and safety

* The original is ALWAYS copied (and verified byte-for-byte) before an in-place modification, regardless of KEEP_ORIGINAL:
  `<dir>/.originals/<name>`; if that exists, `<stem>.<YYYYmmdd-HHMMSS>[-n]<ext>`. With `-o` the input is never modified.
* Output is written to a temp file in the same directory, validated (opens, page count, every page has an image), then
  `os.replace`d atomically. On any exception the original stays untouched, the traceback is logged and the exit code is 1.
* Files that are not plain scans (digital PDFs, OCR'd PDFs, scanpp output with a text layer) are not touched and no backup is made; exit code 0.

## Tests

    .venv/bin/python -m pytest tests -q                 # everything (~1-2 min, depending on machine load)
    .venv/bin/python -m pytest tests -q -m "not slow"   # skip corpus/subprocess/pdftoppm tests (faster)

Needs the `test` extra (`pytest`, `img2pdf`; both are in `requirements.txt`). Corpus-based tests (`test_port_equivalence.py`,
`test_integration.py`) skip themselves when the private `research/` inputs or `pdftoppm` are missing: 20 skips on a fresh
clone, the rest run. Optional tools that enable more tests: `pdftoppm` (poppler-utils), `mutool` (mupdf-tools), `jbig2`
(jbig2enc-tools) and `tesseract`. `test_port_equivalence.py` checks the ported cleaning/geometry functions against the
research prototypes (max abs diff <= 1).

## Research record

[research/README.md](research/README.md) indexes the studies that decided every default, with one report per study and the scripts that
produced the numbers. The corpus pages, scans and most intermediate data are not published, so the scripts need the private inputs to be
re-run; the reports contain aggregate figures only.

## Privacy

scanpp runs entirely locally and sends nothing anywhere. The log (`~/.cache/scanpp/scanpp.log`) holds file names, job ids, timings and
per-page statistics (blank scores, skew, sizes). Page content and OCR text are not logged. With `--ocr` the recognised text is
written only into the output PDF.

## License

MIT, see [LICENSE](LICENSE).
