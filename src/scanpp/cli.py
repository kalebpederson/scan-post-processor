"""Command line: simple-scan post-processing protocol and a normal CLI.

simple-scan:  script MIME_TYPE KEEP_ORIGINAL FILE_PATH [ARGS]     (ARGS is ONE string -> shlex-split)
normal:       scanpp [options] INPUT [INPUT ...] [-o OUTPUT]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
import shlex
import sys
import time
import traceback
from pathlib import Path

from . import __version__, background, safety
from .notify import notify
from .encode import DEFAULT_QUALITY, QUALITY_MAX, QUALITY_MIN, encoder_names
from .options import DEFAULT_BLANK_THRESHOLD, STAGES, Options

log = logging.getLogger("scanpp")

_MIME_RE = re.compile(r"^[a-z]+/[a-z0-9][a-z0-9.+-]*$", re.I)


def _nonneg_float(s: str) -> float:
    try:
        v = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {s!r}") from None
    if not v >= 0 or v == float("inf"):
        raise argparse.ArgumentTypeError("must be a finite number >= 0")
    return v


def _quality(s: str) -> int:
    try:
        v = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {s!r}") from None
    if not QUALITY_MIN <= v <= QUALITY_MAX:
        raise argparse.ArgumentTypeError(f"must be an integer {QUALITY_MIN}..{QUALITY_MAX}")
    return v


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scanpp",
        description="Post-process scanned documents (PDF/JPEG/PNG/WebP): clean, deskew, drop blanks, downsample, "
                    "recompress. Files are modified in place (after a backup) unless -o is given.",
        epilog="simple-scan protocol: scanpp MIME_TYPE true|false FILE_PATH [\"ARGS\"]")
    p.add_argument("inputs", nargs="*", metavar="INPUT", help="files to process")
    p.add_argument("-o", "--output", help="output file (single input only); default: modify input in place")
    p.add_argument("--paper", choices=("flatten", "keep"), default="flatten",
                   help="flatten paper to white (default) or keep a uniform paper tint")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--drop-blank", dest="drop_blank", action="store_true", default=True,
                   help="drop blank pages (default; if ALL pages are blank, all are kept)")
    g.add_argument("--keep-blank", dest="drop_blank", action="store_false", help="keep blank pages")
    p.add_argument("--blank-threshold", type=_nonneg_float, default=DEFAULT_BLANK_THRESHOLD, metavar="N",
                   help="pages scoring below N are treated as blank (default %(default)g). Typical scores: dust / "
                        "bleed-through / scanner shadows 0-10 (up to ~25 on rare dirty edges); a 0.1 in pencil mark "
                        "~5-15; a 0.5 in dark pencil line ~25-50; a lone page number ~15-45; one handwritten word "
                        "~25-100 (light pencil) or 125+ (dark pencil / pen); a 2 in pencil line 100-160; "
                        "'Page 2 of 2' ~160. Lower = keep more pages (0 keeps everything), higher = drop more. "
                        "Each dropped page's score is in the log and in the --report JSON (blank_score)")
    p.add_argument("--duplex", choices=("auto", "off"), default="auto",
                   help="auto (default): in multi-page PDFs, find the two sides of a double-sided sheet and use the side "
                        "that carries coloured marker to remove its bleed-through from the other side (only when "
                        "confident that the two pages are one sheet); off: treat every page on its own")
    p.add_argument("--folds", choices=("auto", "off"), default="auto",
                   help="auto (default): remove the gray crease line a folded sheet leaves (only a faint full-width line at "
                        "about 1/3, 1/2 or 2/3 of the page height; thin strokes lying right on it may be lightened); "
                        "off: leave creases alone")
    p.add_argument("--no-deskew", dest="deskew", action="store_false", help="do not straighten pages")
    p.add_argument("--no-edge-clean", dest="edge_clean", action="store_false",
                   help="do not paint over scanner edge strips / spiral coil")
    p.add_argument("--dpi", type=int, default=None,
                   help="output ppi for the jpeg/jpeg200 encoders only (default 150 / 200; never upsamples). The auto "
                        "and mrc encoders choose their own resolutions (see --quality; at the default level 5: 300 ppi "
                        "mask, 100 ppi colour layer, 150 ppi JPEG pages) and ignore it. For image files (JPEG/PNG/WebP) the default is 200")
    p.add_argument("--jobs", type=int, default=0, help="worker processes (default: CPU count)")
    p.add_argument("--encoder", default="auto", choices=encoder_names(),
                   help="page encoder (the numbers are those of the default --quality 5): auto = per page: no-colour -> 300 ppi JBIG2 ink mask + 100 ppi gray layer; "
                        "highlighter/ink-heavy -> 150 ppi q55 4:4:4 JPEG; otherwise ink mask + 100 ppi colour layer. "
                        "mrc = always mask + colour layer; jpeg = always 150 ppi q55 4:4:4 JPEG; "
                        "jpeg200 = same at 200 ppi (max legibility). Default: auto")
    p.add_argument("--quality", type=_quality, default=DEFAULT_QUALITY, metavar="N",
                   help="output image quality 1..10 (default %(default)d): higher = closer to the cleaned page, bigger files; "
                        "lower = smaller. Controls how the cleaned page is ENCODED only; the cleaning (paper whitening, "
                        "ink gate, deskew, blank dropping) is the same at every level (use --paper keep to keep the paper "
                        "tint). Applies to PDF pages. An explicit --dpi still overrides the ppi of the jpeg/jpeg200 "
                        "encoders; --encoder picks the scheme, --quality its JPEG quality and layer resolution. "
                        "Levels 1-6 keep the 1-bit ink mask schemes; from level 7 on, pages are full-page "
                        "baseline JPEG (4:4:4) at higher ppi/quality (level 10: near-lossless 300 ppi). See README \"Quality\"")
    p.add_argument("--backup-dir", help="where to put backups (default: .originals/ next to the input)")
    p.add_argument("--no-backup", dest="backup", action="store_false", help="do not back up the original (dangerous)")
    p.add_argument("--ocr", action="store_true",
                   help="add an invisible, searchable text layer to PDFs (tesseract; off by default; adds CPU time)")
    p.add_argument("--ocr-lang", default="eng", metavar="LANG",
                   help="tesseract language(s) for --ocr, e.g. eng or eng+spa (default: eng)")
    p.add_argument("--ocr-min-conf", type=float, default=60.0, metavar="N",
                   help="drop OCR words with tesseract confidence below N (0-100; default 60: removes most junk "
                        "from handwriting, costs ~1%% of real words)")
    p.add_argument("-v", "--verbose", action="store_true", help="verbose log and per-stage CPU timing")
    p.add_argument("--report", metavar="FILE.json", help="write per-page statistics as JSON")
    m = p.add_mutually_exclusive_group()
    m.add_argument("--background", dest="background", action="store_true", default=None,
                   help="return immediately and process in a detached, queued worker (default for the simple-scan "
                        "protocol, except for Email temp files)")
    m.add_argument("--foreground", dest="background", action="store_false",
                   help="process before returning (default for the normal CLI)")
    p.add_argument("--no-notify", dest="notify", action="store_false", help="no desktop notifications")
    p.add_argument("--version", action="version", version=f"scanpp {__version__}")
    return p


def is_simple_scan_protocol(argv: list[str]) -> bool:
    return len(argv) >= 3 and bool(_MIME_RE.match(argv[0])) and argv[1] in ("true", "false")


def parse_args(argv: list[str]):
    """Returns (namespace, keep_original or None). Raises SystemExit on usage errors (argparse)."""
    parser = build_parser()
    if is_simple_scan_protocol(argv):
        mime, keep, file_path, *rest = argv
        extra: list[str] = []
        for a in rest:                      # simple-scan passes the whole setting as ONE argument
            extra += shlex.split(a)
        ns = parser.parse_args(extra)
        if ns.inputs or ns.output:
            parser.error("unexpected positional arguments / -o in simple-scan ARGS")
        ns.inputs = [file_path]
        ns.mime = mime
        return ns, keep
    ns = parser.parse_args(argv)
    ns.mime = None
    if not ns.inputs:
        parser.error("no input files")
    if ns.output and len(ns.inputs) != 1:
        parser.error("-o/--output requires exactly one input")
    return ns, None


def options_from(ns) -> Options:
    names = {f.name for f in dataclasses.fields(Options)}
    return Options(**{k: v for k, v in vars(ns).items() if k in names})


def format_timing(results) -> str:
    lines = []
    for r in results:
        if r.status != "processed":
            continue
        tot = sum(r.stage_cpu.values()) or 1e-9
        lines.append(f"{Path(r.path).name}: wall {r.wall:.2f}s, CPU {tot:.1f}s over {r.pages_in} pages "
                     f"({r.pages_in - r.pages_out} dropped), {r.bytes_in / 1e6:.2f} MB -> {r.bytes_out / 1e6:.2f} MB")
        lines.append("  stage CPU: " + ", ".join(
            f"{s} {r.stage_cpu.get(s, 0):.2f}s ({100 * r.stage_cpu.get(s, 0) / tot:.0f}%)" for s in STAGES))
    return "\n".join(lines)


def should_background(ns, protocol: bool) -> bool:
    """Background by default for the simple-scan protocol, foreground for the normal CLI. Always foreground for
    simple-scan's Email temp dir (it attaches the file right after we return), for -o, and for several inputs."""
    if len(ns.inputs) != 1 or ns.output:
        return False
    if background.is_email_tmp(ns.inputs[0]):
        return False
    return protocol if ns.background is None else ns.background


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--_worker"]:               # internal: detached background worker
        return background.run_worker(argv[1])
    ns, keep = parse_args(argv)
    safety.setup_logging(ns.verbose)
    try:
        log.info("scanpp %s argv=%r", __version__, argv)
        if keep is not None:
            log.info("simple-scan protocol: mime=%s KEEP_ORIGINAL=%s file=%s", ns.mime, keep, ns.inputs[0])
        opts = options_from(ns)
        if opts.dpi is not None and opts.dpi < 50:
            raise ValueError("--dpi too small")
        if should_background(ns, keep is not None):
            return background.launch(ns.inputs[0], opts, keep)
        from .pipeline import process_file, write_report
        results = []
        t0 = time.perf_counter()
        for f in ns.inputs:
            if not Path(f).is_file():
                raise FileNotFoundError(f)
            r = process_file(f, opts, output=ns.output, keep_original=keep)
            results.append(r)
            if r.status == "processed":
                log.info("%s: %d -> %d pages, %d -> %d bytes in %.2fs", f, r.pages_in, r.pages_out,
                         r.bytes_in, r.bytes_out, r.wall)
        if ns.verbose:
            print(format_timing(results), file=sys.stderr)
            print(f"total wall time: {time.perf_counter() - t0:.2f}s", file=sys.stderr)
        if ns.report:
            write_report(ns.report, opts, results)
        return 0
    except Exception as e:  # noqa: BLE001  - originals are never replaced before validation
        log.error("FAILED; original left untouched\n%s", traceback.format_exc())
        if keep is not None and getattr(ns, "notify", True):     # simple-scan run: tell the user
            notify("scanpp failed", f"{type(e).__name__}: {e}. Original kept. Log: {safety.log_dir() / 'scanpp.log'}",
                   urgent=True)
        return 1
