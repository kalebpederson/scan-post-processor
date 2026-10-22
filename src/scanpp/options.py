"""Options dataclass (kept import-light: used by the quick foreground path)."""
from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_BLANK_THRESHOLD = 15.0   # blank_score below this -> blank page (calibration: README "Blank pages", research/brother_live/REPORT.md)
STAGES = ("decode", "edge", "blank", "skew", "clean", "deskew", "resize", "encode", "ocr", "duplex_maps", "duplex_supp", "fold_detect", "fold_remove")


@dataclass
class Options:
    paper: str = "flatten"
    drop_blank: bool = True
    blank_threshold: float = DEFAULT_BLANK_THRESHOLD   # pages with geometry.blank_score below this count as blank
    duplex: str = "auto"          # "auto": use the other side of a double-sided sheet against bleed-through (PDFs); "off"
    folds: str = "auto"           # "auto": remove gray fold-crease lines (full-width, faint, at the usual 1/3, 1/2, 2/3 fold rows); "off"
    deskew: bool = True
    edge_clean: bool = True
    dpi: int | None = None        # jpeg/jpeg200 encoders only (None: 150 / 200); image files: 200
    jobs: int = 0                 # 0 -> os.cpu_count()
    encoder: str = "auto"
    quality: int = 5              # 1..10, ENCODING only (5 = historical output); see encode.quality_settings
    backup: bool = True
    backup_dir: str | None = None
    ocr: bool = False
    ocr_lang: str = "eng"
    ocr_min_conf: float = 60.0    # words below this tesseract confidence get no text (see README)
    verbose: bool = False
    report: str | None = None
    notify: bool = True
    background: bool | None = None   # None: auto (background for simple-scan protocol)

    def worker_jobs(self) -> int:
        return self.jobs if self.jobs and self.jobs > 0 else (os.cpu_count() or 1)
