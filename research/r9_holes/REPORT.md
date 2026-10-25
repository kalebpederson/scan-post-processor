# R9 - Punched binding holes

> Figures referenced below are not published (they show private scans). Scripts and aggregate JSON are in this directory; pages are
> keyed by index only.

## Answer

Scans of a book whose binding was cut off (the user's book scans, Brother ADS-2200 sheet feeder, 300 dpi colour, 2549 px wide) carry a
column of punched binding holes along the binding edge (left on fronts, right on backs). The scanner's gray backing shows through each
hole as a solid disc. The existing edge cleanup never touched them (they are detached from the border by a thin white margin) and the
cleaner kept them as gray dots. A new edge rule detects the row and paints exactly the holes with paper colour, before cleaning,
blank-score and deskew. Switch: `geometry.BINDING_HOLES` (False reproduces the previous output exactly).

## 1. Diagnosis

Holes: diameter 0.17 in (52 px at 300 ppi), pitch 0.25 in, centre 0.14-0.2 in from the border, 44 per page (the whole height), a white
paper margin of about 0.06 in between hole and border. Inside: gray 0.65-0.7 x paper, light bluish, with a dark (0.25 x paper) rim along
the top edge from the shadow. The edge logic (`_edge_runs`) needs dark pixels from the border inwards, so the margin stops it; the
sheet-edge-line rule of R7 needs a long straight line. The discs also counted as "sharp ink" in the blank score (gradient at the rim).

## 2. Rule (`geometry._binding_holes`, called from `_edge_mask_work_g` per side, skipped on sides with a coil)

Candidates are 8-connected blobs of pixels darker than 0.88 x paper in the first 0.8 in of a border at the 100 ppi working resolution.
A candidate hole must be

- detached from the border (not touching it), 0.10-0.30 in across in both directions, aspect <= 1.6, filled >= 65 % of its box,
  centre within 0.4 in of the border (keeps hole + rim + 0.03 in inside the 0.6 in edge band);
- an ellipse or a rectangle: IoU with the ellipse of its (opened, specks removed) box >= 0.80, or >= 93 % of the box (rectangular,
  comb / wire holes);
- surrounded by paper: >= 85 % of a 2 px ring around it is > 0.9 x paper (kills hollow bubbles, text, table cells);
- uniform and not paper inside: the core (distance >= half the maximum) has median <= 0.85 x paper and std <= 0.15 x paper.

A row is accepted when >= 6 candidates share the centre depth (+-0.04 in), size (+-20 %), have a regular pitch 0.15-0.6 in and >= 1.2 x
their size (members off the pitch grid by > 0.15 pitch are dropped), and occupy >= 60 % of the pitch slots between the first and last
hole (a few missing or occluded holes are fine). Accepted holes are painted with paper colour dilated by 0.03 in. Two extensions inside an
accepted row: a hole cut by the end of the page is adopted if its depth/size match; a hole fused with a mark that touches it (pen stroke,
lint) fails the shape test, so at an empty pitch slot only the round part (opening with a disc of 0.8 x the hole size, size and
position checked) is painted and the mark is left alone. Nothing outside the holes is painted. Stat: info key `holes_<side>` (count)
and `binding_holes` per page in `--report`. The blank score uses the same mask.

Not handled: standard 3-hole punches. The detector re-run with relaxed constants (>= 3 holes, 0.2-0.45 in, pitch 3.5-5 in,
`probe_3hole.py`) fires on none of the 66 corpus pages or 2,158 sample pages, so there is no example; a 3-hole rule would have to be
anchored on fixed positions to be safe and was not added. Rectangular (comb / wire) binding: the shape test accepts rectangles, but no
example exists in the corpus; covered by the same row logic only in theory.

## 3. Evaluation

(a) The user's book scans (`Bundled-Notes`, 24 pages, plus three separately scanned sections of the same book, 12 pages): 36 of 36 pages
fire, 44 holes each on both sides (L on fronts, R on backs), 44 of 44 holes visible; an independent profile through the hole centres
(`work`-only check) counted 44 on all 24 bundled pages. No hole is left in the cleaned image. Before/after crops reviewed on every page type;
whole pages checked for pages 1-2 and the processed pipeline output for pages 1-4: no handwriting, print or graph touched. A pen mark
touching a hole on one page is left in place, the hole beside it painted. Largest painted area 1.7 % of a page (all holes + the existing
strips). The other 23 raw pages of the book folder (earlier coil scans) do not fire. In the full pipeline `--report` shows
`binding_holes: {L: 44}` or `{R: 44}`, 0 dropped pages.

(b) Corpus (66 pages) and samples (2,158 readable pages, incl. spiral notebooks and graph paper): 0 pages fire on either. Full
pipeline old vs new: corpus 66 of 66 outputs byte-identical; samples 384 processable inputs, 103 byte-identical, 281 differ only in PDF
metadata (dates / ids), 0 pages differ in image bytes or content streams (`ab_pdfcmp.py`); 30 inputs are not processable by either
version (as in R7). First version found one false positive: the bold "2" of a printed power table (a column of 8 identical glyphs at
regular pitch near the margin), painted at the left edge. The shape test (ellipse IoU: glyphs <= 0.73, holes >= 0.83), the
85 % paper ring and the opening for specks came from that; the table is untouched now.

(c) Synthetic negatives (`tests/test_binding_holes.py`, all unchanged and not firing): printed bullets, large black bullets at tight
pitch, bubble-sheet rings, dot leaders, tally marks, a glyph column, a single isolated circle, two short groups of circles far apart,
irregular pitch, 5 holes only, coil teeth, plain text. Positives: left and right rows, top row, 4 missing holes, hole fused with a
pen stroke, switch off restores the old output; only pixels within 0.13 in of a hole centre change.

Tests: 371 passed, 1 skipped (`test_port_equivalence` switches the rule off, like the R7 switches).

## 4. Limitations

- Holes must be detached from the border (a hole cut open to the edge is handled by the strip rule only if dark and attached).
- A row of 6 or more identical, uniform, backing-filled, paper-surrounded discs or rectangles near a border is treated as holes whatever
  they are (a printed column of solid circles of 0.1-0.3 in, regularly spaced, would be painted). None in the corpus.
- Holes whose interior is lighter than 0.85 x paper (a white backing) are not found.
- Holes on pages with a coil on the same side are left to the coil rule; only the page-end clipping of the very last hole is adopted when
  the row is established first.
- No 3-hole punch support (see above).
