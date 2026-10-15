"""Hand-picked labelled regions (native pixel coords [x,y,w,h]).  pos = faint REAL content; neg = bleed-through only."""
import json
C = "research/corpus/"; E = "research/scans/epson_300dpi.png"; PG = "research/r2_cleaning/work/pages/"
pos = [
 dict(id="pencil_pair03_a", cls="pencil", file=C+"pair03_front.jpg", dpi=300, box=[1215, 868, 300, 50], dmax=0.9),
 dict(id="pencil_pair03_b", cls="pencil", file=C+"pair03_front.jpg", dpi=300, box=[1200, 1085, 220, 85], dmax=0.9),
 dict(id="pencil_image04", cls="pencil", file=C+"image04.jpg", dpi=200, box=[110, 365, 520, 65], dmax=0.9),
 dict(id="pencil_epson_hatch", cls="pencil", file=E, dpi=300, box=[1030, 440, 110, 240], dmax=0.9),
 dict(id="grid_image04_a", cls="grid", file=C+"image04.jpg", dpi=200, box=[975, 288, 125, 60], dmax=0.9),
 dict(id="grid_image04_b", cls="grid", file=C+"image04.jpg", dpi=200, box=[985, 812, 165, 138], dmax=0.9),
 dict(id="grid_pair07", cls="grid", file=C+"pair07_front.jpg", dpi=150, box=[190, 1000, 150, 130], dmax=0.9),
 dict(id="grid_single28", cls="grid", file=C+"single28.jpg", dpi=150, box=[683, 278, 139, 89], dmax=0.9),
 dict(id="ruled_epson", cls="ruled", file=E, dpi=300, box=[640, 770, 780, 110], dmax=0.9),
 dict(id="hl_cyan_single09", cls="highlighter", file=C+"single09.jpg", dpi=150, box=[345, 1165, 295, 17], dmax=0.9),
 dict(id="hl_cyan_pair06", cls="highlighter", file=C+"pair06_front.jpg", dpi=200, box=[1330, 886, 260, 28], dmax=0.5),
 dict(id="hl_yellow_pair06", cls="highlighter", file=C+"pair06_front.jpg", dpi=200, box=[1286, 740, 18, 130], dmax=0.9),
 dict(id="grayline_pair06", cls="light_print", file=C+"pair06_front.jpg", dpi=200, box=[1585, 700, 23, 240], dmax=0.9),
 dict(id="dotted_epson_text", cls="light_print", file=E, dpi=300, box=[1495, 480, 45, 265], dmax=0.9),
 dict(id="gray_text_epson", cls="light_print", file=E, dpi=300, box=[2200, 220, 45, 310], dmax=0.9),
 dict(id="dotted_epson_rules", cls="light_print", file=E, dpi=300, box=[1810, 180, 150, 80], dmax=0.9),
 dict(id="lightblue_pen_single06", cls="light_pen", file=C+"single06.jpg", dpi=200, box=[1000, 570, 180, 70], dmax=0.9),

 dict(id="grid_pink_paper", cls="grid", file=PG+"390e6aba/pg-002.jpg", dpi=200, box=[880, 880, 120, 120], dmax=0.9),
]
neg_src = [("9e0ae637/pg-001", 200, 1300, 1450), ("c26e6e0c/pg-000", 300, 600, 2925), ("13825b8a/pg-001", 300, 525, 2850),
           ("38d5b2f0/pg-000", 300, 1875, 1950), ("38d5b2f0/pg-003", 300, 1500, 2775), ("7695ba11/pg-002", 150, 296, 1184),
           ("101df960/pg-002", 150, 592, 1443), ("b842f998/pg-001", 150, 1036, 1221), ("6060e859/pg-001", 300, 900, 2325),
           ("101df960/pg-003", 150, 333, 888)]
neg = [dict(id="bleed_%02d" % i, cls="bleed", file=PG + f + ".jpg", dpi=d, box=[x, y, d, d // 2], dmax=0.9) for i, (f, d, x, y) in enumerate(neg_src)]
neg.append(dict(id='bleed_pink', cls='bleed', file=PG+'390e6aba/pg-002.jpg', dpi=200, box=[1180, 1380, 340, 200], dmax=0.9))
json.dump(dict(pos=pos, neg=neg), open("labels.json", "w"), indent=1)
print(len(pos), len(neg))
