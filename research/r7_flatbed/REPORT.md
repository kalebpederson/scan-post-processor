# R7 — Flatbed notebook edges (sheet-edge line, coil ends, border hairline, corner dots)

> Figures referenced below are not published (they show private scans).

(Transcribed by the orchestrator from the R7 agent's final report. Scripts and aggregate JSON are in this directory; pages are keyed by
index only.)

## Answer

Flatbed scans of a spiral notebook (the user's notebook scans, 300 dpi, 2544 px wide) left four edge artefacts that the sheet-feeder-tuned
edge cleanup (R4, paper boundary in the acceptance report) missed. Four conservative rules now paint them with paper colour, before
cleaning. Each has a switch that restores the previous behaviour exactly: `boundary.EDGE_LINE`, `geometry.FLATBED_COIL_FRINGE`,
`FLATBED_CORNER_DOTS`, `FLATBED_HAIRLINE`.

## 1. Diagnosis

Bottom edge of a flatbed scan, from the border inwards: dark scanner frame, 0.05-0.1 in of white lid, the sheet's own edge as a faint grey
line, then paper.

1. `boundary.detect_side` needs non-paper backing from the border to the sheet edge; the white lid in between stops it. The strip logic
   paints only dark runs attached to the border, so the detached line survived and the cleaner then darkened it.
2. The dark-strip threshold (0.75 x paper) misses the frame's light penumbra row along the border.
3. At the coil's ends the paper lifts: a grey/lilac shadow patch and a thin bluish cover line lie beyond the painted band (coil depth +
   0.06 in).
4. In a corner, the strip runs of both borders are each "too long" for the other's painting, so a dark corner block survives.

## 2. Rules

**Detached edge line** (`boundary.detect_edge_line`, only where `detect_side` did not accept the side): paints border-to-line plus the line
and 0.02 in, when all hold: line 0.04-0.3 in from the border, tilt <= 3 deg, thickness <= 0.12 in; mean darkness 6-40 levels below paper
(faint, not a printed black rule); present in >= 90 % of the 0.4 in windows along the whole side (coil band excluded); a dark scanner strip
hugs >= 40 % of the border; the lid between strip and line is blank (< 1 % ink-like pixels); on coloured paper the lid must be white.

**Coil-end fringe** (`_coil_fringe`, coil pages only): within 0.9 in of the page ends, pixels connected to the painted band are added when
they are (a) light unsaturated shading in blobs >= 0.05 in thick within 0.25 in of the band, or (b) a >= 0.15 in run of saturated pixels
within 0.07 in of the top/bottom border and 0.55 in of the band. Thin or dark strokes and highlighter are excluded.

**Border hairline** (`_border_hairline`, coil pages only, borders without a coil): paints at most the first 0.04 in of a border when >= 70 %
of it is > 12 levels below paper (or already painted) and >= 30 % is shaded.

**Corner dots** (`_corner_dots`): a dark blob touching an image border, within 0.15 in of both borders and touching the painted strip mask.

Known limit: a faint, full-width printed rule within 0.3 in of the border with a scanner frame beyond it is indistinguishable from a sheet
edge and is erased.

## 3. Evaluation

| set | pages | pages changed at edge level | line / coil fringe / corner / hairline (overlapping) | pipeline outputs |
|---|---|---|---|---|
| corpus | 66 | 35 | 1 / 9 / 4 / 29 | 32 byte-identical, 34 differ |
| samples | 2,158 readable | 650 | 9 / 236 / 166 / 456 | 83 identical, 43 metadata-only, 253 differ (502 pages), 35 not processable by either version |

- Every output that differs has an edge rule firing; no page differs without one (corpus and samples).
- The detached-line rule fires on 10 pages; all 10 inspected: genuine sheet-edge lines with lid and frame beyond, nothing printed removed.
- Largest painted area 0.7 % of a page. Reviewed by eye: all 35 corpus pages, all line firings, 32 random corner-only pages, 99 random
  sample before/after/diff tiles. No text, rule, grid, handwriting, highlighter or printed corner was erased. Scattered whole-page pixel
  differences on some pages are sub-pixel deskew/cleaner noise from the slightly different input. About 400 changed sample pages were
  not rendered.
- An earlier version fired on printed box tops at 0.48 in on yellow pages; hence the 0.3 in depth cap, the darkness cap and the white-lid
  check on coloured paper.
- Synthetic negatives (`tests/test_flatbed_edge.py`): printed rules 0.2 / 0.35 / 0.5 in from the bottom with a frame beyond, footer
  rule, table border near the edge, handwriting along the edge, thick black rule, short rule, text on the lid, faint line without a
  frame — all unchanged.

## 4. The user's notebook scans

Sheet-edge line above the bottom edge gone; frame penumbra rows removed; coil-end shadow patch and bluish cover line (top-right of a back
page) gone; black corner dot gone; real paper marks near the edge untouched. One page keeps a partial single-pixel row at the very bottom.

## 5. Open

Two sample PDFs have a MediaBox implying 831 dpi, which mis-scales all edge logic; this predates R7 and is not investigated here.
