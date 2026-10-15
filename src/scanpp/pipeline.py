"""Per-page processing and parallel orchestration."""
from __future__ import annotations

import dataclasses
import io
import logging
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pikepdf
from PIL import Image

from . import duplex, fold, geometry, imageio, ocr, pdfio, safety
from .clean import clean_page, clean_page_reference
from .encode import EncodedPage, encode_page

log = logging.getLogger("scanpp")

from .options import STAGES, Options  # noqa: E402,F401

SMALL_IMAGE_IN = 3.0       # images smaller than this (inches, either side) fall back to the reference cleaner on failure
IMAGE_FILE_DPI = 200      # output ppi of JPEG/PNG/WebP inputs (--dpi default for those)
DUPLEX_MAX_REMOVED = 0.25   # a duplex mask that would remove more than this share of a page's ink is discarded (observed max on real pairs: 0.083)


# ----------------------------------------------------------------------------- fold creases
@dataclass
class FoldHint:
    """Fold evidence computed ahead by the duplex stage 0 (which decodes and edge-cleans every page anyway): this page's stage-1 line
    candidates (`own`, CPU seconds `t`) and those of its sheet partner (`partner`, tier B)."""
    own: list | None = None
    partner: list | None = None
    t: float = 0.0


def _fold_hint(hints: dict | None, k: int, plan: DuplexPlan | None) -> FoldHint | None:
    own = (hints or {}).get(k)
    if own is None:
        return None
    other = (hints or {}).get(plan.partner) if plan is not None else None
    return FoldHint(own[0], other[0] if other else None, own[1])


def _remove_folds(img: np.ndarray, dpi: float, stats: dict | None, tick, hint: FoldHint | None = None, src: bool = False):
    """Detect fold creases on the edge-cleaned upright page and remove them in place.  Returns (img, changed); any failure leaves the page as it
    is.  `src`: the duplex partner page being prepared as a bleed-through source (same decision as for the page itself; no stats)."""
    try:
        t0 = time.process_time()
        folds = fold.detect_folds(img, dpi, partner=hint.partner if hint else None, cands=hint.own if hint else None)
        if not src:
            tick("fold_detect", t0)
        if not folds:
            return img, False
        t0 = time.process_time()
        if not img.flags.writeable:
            img = img.copy()
        mask = fold.fold_mask(img, dpi, folds)
        fold.remove_folds(img, dpi, folds, mask=mask, inplace=True)
        if not src:
            stats["folds"] = [{"row_frac": round(f.y_frac, 4), "angle": round(f.angle_deg, 2), "tier": f.info["tier"],
                               "score": round(f.info["score"], 3)} for f in folds]
            stats["fold_px_removed"] = int(mask.sum())
            tick("fold_remove", t0)
        return img, True
    except Exception as e:  # noqa: BLE001
        log.warning("fold removal failed (%s: %s); page processed without it", type(e).__name__, e)
        return img, False


def _log_folds(label: str, stats: dict) -> None:
    if stats.get("folds"):
        log.info("%s: removed %d fold crease(s) (%s), %d px", label, len(stats["folds"]),
                 ", ".join(f"{f['row_frac']:.3f} tier {f['tier']}" for f in stats["folds"]), stats["fold_px_removed"])


# ----------------------------------------------------------------------------- core (pure, array in / array out)
def process_array(rgb: np.ndarray, native_dpi: float, opts: Options, allow_drop: bool = True, resize: bool = True, post_clean=None,
                  fold_hint: FoldHint | None = None):
    """Run the decided pipeline on an upright RGB page.

    `post_clean(img, out, stats)` (optional) runs between the cleaner and deskew, on the edge-cleaned page and its cleaned version, and
    returns the cleaned page to continue with (the duplex bleed-through suppression; never called for a page dropped as blank).

    Returns (rgb_out or None if dropped as blank, out_dpi, stats). Stats has per-stage CPU seconds in stats['t'].
    `fold_hint` (optional): fold evidence computed ahead (duplex stage 0), see `FoldHint`.
    NOTE: if `rgb` is writeable it is modified in place by edge cleaning and fold removal (callers pass a private decode buffer).
    """
    t: dict[str, float] = {}
    stats: dict = {"native_dpi": round(native_dpi, 2), "blank_score": None, "dropped": False,
                   "skew_angle": None, "skew_conf": None, "skew_applied": False, "edge_painted_frac": 0.0,
                   "duplex_partner": None, "duplex_ncc": None, "duplex_removed_px": 0, "folds": [], "fold_px_removed": 0}

    def tick(name, t0):
        t[name] = t.get(name, 0.0) + time.process_time() - t0

    img = rgb
    gray = wdpi = None
    if opts.edge_clean:
        t0 = time.process_time()
        img, info, gray, wdpi = geometry._clean_edges_g(img, native_dpi, inplace=bool(img.flags.writeable))
        stats["edge_painted_frac"] = info["painted_frac"]
        if bs := {k[-1]: round(v, 3) for k, v in info.items() if k.startswith("boundary_")}:
            stats["paper_boundary_in"] = bs               # side -> depth (inches) of the detected paper edge
        tick("edge", t0)

    if opts.folds == "auto":
        if fold_hint is not None and fold_hint.t:
            t["fold_detect"] = fold_hint.t                    # (the stage-1 part was done by the duplex stage 0)
        img, changed = _remove_folds(img, native_dpi, stats, tick, fold_hint)
        if changed:
            gray = None                                       # blank score, skew and the cleaner see the de-creased page

    t0 = time.process_time()
    if gray is None:       # (an unpainted page: the edge stage already made it)
        gray, wdpi = geometry.prep_gray(img, native_dpi)     # one work-resolution (<=100 ppi) gray copy for blank+skew
    score = geometry.blank_score(gray, wdpi)
    stats["blank_score"] = round(score, 2)
    blank = score < opts.blank_threshold
    stats["blank"] = bool(blank)
    tick("blank", t0)
    if blank and opts.drop_blank and allow_drop:
        stats["dropped"] = True
        stats["t"] = t
        return None, native_dpi, stats

    angle, conf = 0.0, 0.0
    if opts.deskew and not blank:
        t0 = time.process_time()
        angle, conf = geometry.estimate_skew(gray, wdpi)
        tick("skew", t0)
        stats["skew_angle"], stats["skew_conf"] = round(angle, 3), round(conf, 3)
    del gray

    t0 = time.process_time()
    try:
        out = clean_page(img, native_dpi, paper=opts.paper)
    except Exception as e:  # noqa: BLE001
        if min(img.shape[:2]) >= SMALL_IMAGE_IN * native_dpi:
            raise                                          # a real page: a failure is a failure
        # tiny image: the optimised path has size corner cases; the reference implementation has none
        log.warning("clean_page failed (%s: %s); using the reference implementation", type(e).__name__, e)
        stats["clean_fallback"] = True
        out = clean_page_reference(img, native_dpi, paper=opts.paper)
    tick("clean", t0)

    if post_clean is not None:
        t0 = time.process_time()
        out = post_clean(img, out, stats)
        tick("duplex_supp", t0)

    if opts.deskew and not blank and geometry.should_deskew(angle, conf):
        t0 = time.process_time()
        fill = (255, 255, 255) if opts.paper == "flatten" else geometry.paper_colour(out)
        out = geometry.deskew(out, angle, fill)
        stats["skew_applied"] = True
        tick("deskew", t0)

    t0 = time.process_time()
    out_dpi = native_dpi
    img_dpi = opts.dpi or IMAGE_FILE_DPI          # PDF pages are resampled by the encoders (resize=False)
    if resize and img_dpi < native_dpi * 0.98:
        s = img_dpi / native_dpi
        h, w = out.shape[:2]
        out = cv2.resize(out, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
        out_dpi = float(img_dpi)
    tick("resize", t0)
    stats["t"] = t
    stats["out_dpi"] = round(out_dpi, 2)
    return out, out_dpi, stats


# ----------------------------------------------------------------------------- PDF page task (runs in workers)
def _decode(task: pdfio.PageInput) -> np.ndarray:
    if task.kind == "raw":
        w, h = task.size
        return np.frombuffer(task.data, np.uint8).reshape(h, w, 3).copy()
    # libjpeg-turbo via OpenCV is ~1.6x faster than PIL's decode + convert + copy (pixel-identical for baseline JPEG);
    # the result is a writeable private buffer.  Anything unexpected falls back to PIL.
    arr = cv2.imdecode(np.frombuffer(task.data, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if arr is not None and arr.ndim == 3 and arr.shape[2] == 3 and (arr.shape[1], arr.shape[0]) == tuple(task.size):
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    with Image.open(io.BytesIO(task.data)) as im:
        return np.array(im.convert("RGB"))


# ----------------------------------------------------------------------------- duplex (bleed-through from the other side of the sheet)
@dataclass
class DuplexPlan:
    """Stage-0 result for one page: its sheet partner and the registration partner -> this page."""
    partner: int
    reg: duplex.PairReg
    ncc: float


def _upright_edge_clean(task: pdfio.PageInput, opts: Options) -> np.ndarray:
    """Decode a page the way process_pdf_page presents it to the cleaner: upright and (if enabled) edge-cleaned."""
    rgb = _decode(task)
    if task.rotate:
        rgb = np.ascontiguousarray(np.rot90(rgb, -task.rotate // 90))
    if opts.edge_clean:
        rgb = geometry._clean_edges_g(rgb, task.dpi, inplace=True)[0]
    return rgb


def _duplex_maps(task: pdfio.PageInput, opts: Options):
    """Stage 0 (worker): 50 ppi ink maps of one page, or None if it cannot be read, and (folds on) its fold line candidates with their CPU
    seconds (the page is decoded here anyway; the sheet partner's candidates are tier-B evidence).  Returns (index, maps, cpu seconds,
    (candidates, cpu seconds) or None); the cpu seconds of the maps exclude the fold part."""
    t0 = time.process_time()
    fc = None
    try:
        rgb = _upright_edge_clean(task, opts)
        maps = duplex.ink_maps(rgb, task.dpi)
        if opts.folds == "auto":
            t1 = time.process_time()
            try:
                fc = (fold.line_candidates(rgb, task.dpi)[0], 0.0)
            except Exception as e:  # noqa: BLE001
                log.warning("page %d: fold analysis failed (%s: %s)", task.index + 1, type(e).__name__, e)
            else:
                fc = (fc[0], time.process_time() - t1)
    except Exception as e:  # noqa: BLE001
        log.warning("page %d: duplex analysis failed (%s: %s); page not paired", task.index + 1, type(e).__name__, e)
        maps = None
    return task.index, maps, time.process_time() - t0 - (fc[1] if fc else 0.0), fc


def _map_pool(fn, inputs, jobs: int, *args) -> list:
    """fn(task, *args) for each task, in task order; at most 2*jobs tasks in flight (inline for one job)."""
    if jobs <= 1:
        return [fn(task, *args) for task in inputs]
    out: list = []
    pending: deque = deque()
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker) as ex:
        for task in inputs:
            pending.append(ex.submit(fn, task, *args))
            del task
            while len(pending) >= jobs * 2:
                out.append(pending.popleft().result())
        while pending:
            out.append(pending.popleft().result())
    return out


def plan_duplex(inputs, n_pages: int, opts: Options, name: str = "") -> tuple[dict, dict]:
    """Stage 0: find the front/back sheets of a PDF.  Returns ({page index: DuplexPlan}, {page index: cpu seconds of its analysis}).
    The parent keeps only the small 50 ppi maps; the workers decode the pages."""
    return _plan_duplex(inputs, n_pages, opts, name)[:2]


def _plan_duplex(inputs, n_pages: int, opts: Options, name: str = "") -> tuple[dict, dict, dict]:
    """`plan_duplex` + {page index: (fold line candidates, cpu seconds)}."""
    if opts.duplex != "auto" or n_pages < 2:
        return {}, {}, {}
    jobs = min(opts.worker_jobs(), n_pages)
    try:
        got = _map_pool(_duplex_maps, inputs, jobs, opts)
        maps = [m for _, m, _, _ in got]
        times = {i: t for i, _, t, _ in got}
        hints = {i: fc for i, _, _, fc in got if fc is not None}
        plan: dict = {}
        for i, j, f in duplex.pair_sheets(maps):
            reg = f["reg"]            # page i -> page j
            plan[j] = DuplexPlan(i, reg, f["ncc"])
            plan[i] = DuplexPlan(j, reg.inverse(), f["ncc"])
            log.info("%s: pages %d and %d are the two sides of one sheet (ncc %.2f, peak gaps %.2f/%.2f); suppressing bleed-through",
                     name, i + 1, j + 1, f["ncc"], f["ga"], f["gb"])
        return plan, times, hints
    except Exception as e:  # noqa: BLE001
        log.warning("%s: duplex analysis failed (%s: %s); processing pages independently", name, type(e).__name__, e)
        return {}, {}, {}


def _duplex_post(partner_task: pdfio.PageInput, plan: DuplexPlan, opts: Options, page: int, hint: FoldHint | None = None):
    """Build the `post_clean` hook of one page: remove the partner's bleed-through from the cleaned page.  Any failure leaves the page as is."""
    def post(img, out, stats):
        try:
            src = _upright_edge_clean(partner_task, opts)
            if opts.folds == "auto" and hint is not None and hint.partner is not None:     # the partner's crease is not a source of bleed-through
                src, _ = _remove_folds(src, partner_task.dpi, None, None, FoldHint(hint.partner, hint.own), src=True)
            paper = (255, 255, 255) if opts.paper == "flatten" else geometry.paper_colour(out)
            s = duplex.bleed_suppression(src, img, plan.reg, cleaned=out, paper_rgb=paper)
            del src
            new = duplex.apply_suppression(out, s, paper)
            n_gone = 0
            if new is not out:
                ink = np.abs(out.astype(np.int16) - np.array(paper, np.int16)).max(-1) > 5
                gone = ink & (np.abs(new.astype(np.int16) - np.array(paper, np.int16)).max(-1) <= 5)
                n_ink, n_gone = int(ink.sum()), int(gone.sum())
                if n_gone > DUPLEX_MAX_REMOVED * max(n_ink, 1):
                    log.info("page %d: duplex suppression would remove %d of %d ink px (> %d%%); discarded",
                             page, n_gone, n_ink, round(100 * DUPLEX_MAX_REMOVED))
                    stats["duplex_discarded_px"] = n_gone
                    return out
                out = new
            stats["duplex_partner"] = plan.partner
            stats["duplex_ncc"] = round(plan.ncc, 3)
            stats["duplex_removed_px"] = n_gone
        except Exception as e:  # noqa: BLE001
            log.warning("page %d: duplex suppression failed (%s: %s); page processed without it", page, type(e).__name__, e)
            stats["duplex_error"] = f"{type(e).__name__}: {e}"
        return out
    return post


def _plan_pair(plan: dict, i: int, f: dict, name: str) -> None:
    reg = f["reg"]            # page i -> page i+1
    plan[i + 1] = DuplexPlan(i, reg, f["ncc"])
    plan[i] = DuplexPlan(i + 1, reg.inverse(), f["ncc"])
    log.info("%s: pages %d and %d are the two sides of one sheet (ncc %.2f, peak gaps %.2f/%.2f); suppressing bleed-through",
             name, i + 1, i + 2, f["ncc"], f["ga"], f["gb"])


@dataclass
class PageResult:
    index: int
    encoded: EncodedPage | None
    stats: dict = field(default_factory=dict)


def process_pdf_page(task: pdfio.PageInput, opts: Options, allow_drop: bool = True, partner: pdfio.PageInput | None = None,
                     plan: DuplexPlan | None = None, hint: FoldHint | None = None) -> PageResult:
    t0 = time.process_time()
    rgb = _decode(task)
    t_dec = time.process_time() - t0
    rot = task.rotate
    if rot:                                    # make the page upright for analysis (rotate clockwise by /Rotate)
        rgb = np.ascontiguousarray(np.rot90(rgb, -rot // 90))
    post = _duplex_post(partner, plan, opts, task.index + 1, hint) if partner is not None and plan is not None else None
    out, out_dpi, stats = process_array(rgb, task.dpi, opts, allow_drop, resize=False, post_clean=post, fold_hint=hint)
    del rgb
    _log_folds(f"page {task.index + 1}", stats)
    stats["t"]["decode"] = t_dec
    stats["page"] = task.index + 1
    stats["rotate"] = rot
    if out is None:
        return PageResult(task.index, None, stats)
    layer = None
    if opts.ocr and not stats.get("blank"):
        t0 = time.process_time()
        gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)          # upright, cleaned, deskewed, full resolution
        layer, ostats = ocr.ocr_page(gray, task.dpi, opts.ocr_lang, opts.ocr_min_conf)
        del gray
        stats["ocr"] = ostats
        stats["t"]["ocr"] = time.process_time() - t0 + ostats.get("cpu", 0.0)
    if rot:
        out = np.ascontiguousarray(np.rot90(out, rot // 90))
    bw, bh = task.media_box[2] - task.media_box[0], task.media_box[3] - task.media_box[1]
    t0 = time.process_time()
    ep = encode_page(out, out_dpi, (bw, bh), opts.encoder, opts.dpi)
    stats["t"]["encode"] = time.process_time() - t0
    ep.media_box = task.media_box
    ep.rotate = rot
    ep.ocr = layer
    stats["bytes"] = ep.nbytes
    stats["gray"] = ep.info.get("gray")
    stats["scheme"] = ep.info.get("scheme")
    stats["encoding"] = ep.info
    stats["out_px"] = [int(out.shape[1]), int(out.shape[0])]
    return PageResult(task.index, ep, stats)


def _tune_malloc() -> None:
    """glibc returns every large (>= 128 KiB, growing to 32 MiB) block to the kernel on free and faults it in again on the next allocation; a page
    pass allocates dozens of 4-60 MB planes, so the kernel zeroing / page-fault traffic is a visible share of the (memory-bound) run time.
    Keeping freed memory in the process (bounded: 32 MiB mmap threshold, trim only above 512 MiB) removes it.  No-op off glibc."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.mallopt(-3, 32 * 1024 * 1024)       # M_MMAP_THRESHOLD
        libc.mallopt(-1, 512 * 1024 * 1024)      # M_TRIM_THRESHOLD
        libc.mallopt(-2, 64 * 1024 * 1024)       # M_TOP_PAD
    except Exception:  # noqa: BLE001
        pass


def _init_worker() -> None:
    cv2.setNumThreads(1)
    _tune_malloc()


# ----------------------------------------------------------------------------- orchestration
def run_pages(inputs, n_pages: int, opts: Options, allow_drop: bool = True, plan: dict | None = None, fetch=None,
              hints: dict | None = None) -> list[PageResult]:
    """Process pages (iterable of PageInput, lazily produced) in parallel; results in page order.
    At most 2*jobs pages are in flight, so parent memory stays bounded.
    `plan` (page index -> DuplexPlan) and `fetch` (page index -> PageInput) enable duplex suppression: a planned page is sent together with
    its partner's (compressed) input and the worker decodes both.  `hints`: page index -> (fold line candidates, cpu seconds) from stage 0."""
    jobs = min(opts.worker_jobs(), n_pages)
    plan = plan or {}

    def one(task):
        pl = plan.get(task.index)
        h = _fold_hint(hints, task.index, pl)
        return process_pdf_page(task, opts, allow_drop, fetch(pl.partner), pl, h) if pl is not None else process_pdf_page(task, opts, allow_drop, hint=h)

    if jobs <= 1:
        return [one(task) for task in inputs]
    results: list[PageResult] = []
    window = jobs * 2
    pending: deque = deque()
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker) as ex:
        for task in inputs:
            pl = plan.get(task.index)
            h = _fold_hint(hints, task.index, pl)
            if pl is not None:
                pending.append(ex.submit(process_pdf_page, task, opts, allow_drop, fetch(pl.partner), pl, h))
            else:
                pending.append(ex.submit(process_pdf_page, task, opts, allow_drop, hint=h))
            del task
            while len(pending) >= window:
                results.append(pending.popleft().result())
        while pending:
            results.append(pending.popleft().result())
    return results


def run_pages_duplex(make_inputs, n_pages: int, opts: Options, fetch, name: str = "", allow_drop: bool = True):
    """`plan_duplex` + `run_pages` as one pipeline in one worker pool: the 50 ppi maps are computed ahead by the workers, the parent tests each
    consecutive pair as its maps arrive, and page k is submitted as soon as its sheet is decided, which needs the pair tests up to (k+1, k+2)
    (`duplex.keeps_pair` looks one pair to each side) - so the analysis overlaps the page processing instead of preceding it.  The decisions
    are those of `duplex.pair_sheets`.  `make_inputs()` gives a fresh iterator over the PageInputs.
    Returns (results, plan, {page index: cpu seconds of its analysis}, {page index: (fold line candidates, cpu seconds)})."""
    jobs = min(opts.worker_jobs(), n_pages)
    if opts.duplex != "auto" or n_pages < 2 or jobs <= 1:
        plan, t_maps, hints = _plan_duplex(make_inputs(), n_pages, opts, name)
        return run_pages(make_inputs(), n_pages, opts, allow_drop, plan, fetch, hints), plan, t_maps, hints
    window = jobs * 2
    plan: dict = {}
    t_maps: dict = {}
    hints: dict = {}
    cand: dict = {}                 # i -> features of the accepted pair (i, i+1)
    results: list[PageResult] = []
    pending: deque = deque()        # page tasks
    maps_pending: deque = deque()   # map tasks, in page order
    analysed = 0                    # pages whose maps have been collected
    prev = None
    ok = True
    map_inputs = make_inputs()

    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker) as ex:
        def analyse_upto(n_done: int) -> None:
            nonlocal analysed, prev, ok
            while ok and analysed < min(n_done, n_pages):
                while len(maps_pending) < window and (mt := next(map_inputs, None)) is not None:
                    maps_pending.append(ex.submit(_duplex_maps, mt, opts))
                    del mt
                try:
                    i, m, t, fc = maps_pending.popleft().result()
                    t_maps[i] = t
                    if fc is not None:
                        hints[i] = fc
                    if analysed and (f := duplex.test_pair(prev, m)) is not None:
                        cand[analysed - 1] = f
                    prev = m
                    analysed += 1
                except Exception as e:  # noqa: BLE001
                    log.warning("%s: duplex analysis failed (%s: %s); processing the remaining pages independently", name, type(e).__name__, e)
                    ok = False
                    for fut in maps_pending:
                        fut.cancel()

        for task in make_inputs():
            k = task.index
            analyse_upto(k + 3)
            if ok and k in cand and duplex.keeps_pair(cand, k):       # (pair k-1 was decided with page k-1)
                _plan_pair(plan, k, cand[k], name)
            pl = plan.get(k)
            h = _fold_hint(hints, k, pl)
            if pl is not None:
                pending.append(ex.submit(process_pdf_page, task, opts, allow_drop, fetch(pl.partner), pl, h))
            else:
                pending.append(ex.submit(process_pdf_page, task, opts, allow_drop, hint=h))
            del task
            while len(pending) >= window:
                results.append(pending.popleft().result())
        while pending:
            results.append(pending.popleft().result())
    return results, plan, t_maps, hints


@dataclass
class FileResult:
    path: str
    status: str                        # "processed" | "skipped"
    reason: str = ""
    output: str | None = None
    backup: str | None = None
    pages_in: int = 0
    pages_out: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    wall: float = 0.0
    stage_cpu: dict = field(default_factory=dict)
    pages: list = field(default_factory=list)
    note: str = ""                    # e.g. "target changed; wrote sidecar"


def _sum_stages(stats_list) -> dict:
    tot = {s: 0.0 for s in STAGES}
    for st in stats_list:
        for k, v in st.get("t", {}).items():
            tot[k] = tot.get(k, 0.0) + v
    return {k: round(v, 3) for k, v in tot.items()}


def _prepare_target(path: Path, output: Path | None, opts: Options, keep_original: str | None) -> Path | None:
    """Back up the original if we are going to overwrite it. Returns the backup path (or None)."""
    if output is not None and Path(output).resolve() != path.resolve():
        return None
    log.info("KEEP_ORIGINAL=%s (ignored: originals are always backed up unless --no-backup)", keep_original)
    if not opts.backup:
        log.warning("backup disabled: %s will be overwritten without a copy", path)
        return None
    b = safety.backup_original(path, Path(opts.backup_dir) if opts.backup_dir else None)
    log.info("backed up %s -> %s", path, b)
    return b


def sidecar_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.scanpp{path.suffix}")


def _install(tmp: Path, target: Path, path: Path, fingerprint: dict | None, res: FileResult) -> Path | None:
    """Atomically move the validated `tmp` into place. For queued (background) jobs, first re-check that the
    live target is still the file we enqueued: if it changed, write a sidecar instead; if it vanished, discard."""
    if fingerprint is not None and target == path:
        if not path.exists():
            tmp.unlink(missing_ok=True)
            res.status, res.reason = "skipped", "target was deleted while processing"
            res.note = res.reason
            log.warning("%s: %s; result discarded", path, res.reason)
            return None
        if not safety.fingerprint_matches(fingerprint, path):
            target = sidecar_path(path)
            res.note = f"target changed while processing; result written to {target.name}"
            log.warning("%s: %s", path, res.note)
    safety.atomic_replace(tmp, target)
    return target


def process_pdf_file(path: Path, output: Path | None, opts: Options, keep_original: str | None = None,
                     source: Path | None = None, backup_done: bool = False,
                     fingerprint: dict | None = None) -> FileResult:
    path = Path(path)
    src = Path(source) if source else path
    target = Path(output) if output else path
    t_wall = time.perf_counter()
    res = FileResult(str(path), "processed", bytes_in=src.stat().st_size)
    with pikepdf.open(str(src)) as pdf:
        ok, why = pdfio.classify_pdf(pdf)
        if not ok:
            log.info("%s: not a plain scan (%s); leaving untouched", path, why)
            res.status, res.reason = "skipped", why
            return res
        n = len(pdf.pages)
        res.pages_in = n
        if opts.ocr and (why_not := ocr.check_available(opts.ocr_lang)):
            log.warning("%s: OCR disabled for this file: %s", path, why_not)
            opts = dataclasses.replace(opts, ocr=False)
        if not backup_done:
            res.backup = str(p) if (p := _prepare_target(path, output, opts, keep_original)) else None
        src_info = {k: v for k, v in pdf.docinfo.items()} if pdf.docinfo is not None else {}
        fetch = lambda i: pdfio.extract_page_input(pdf.pages[i], i, path.name)  # noqa: E731
        results, plan, t_maps, hints = run_pages_duplex(lambda: pdfio.iter_page_inputs(pdf, name=path.name), n, opts, fetch, path.name)
        for r in results:
            if (t := t_maps.get(r.index)) is not None:
                r.stats["t"]["duplex_maps"] = t
        for r in results:
            if r.encoded is None:
                log.info("%s: page %d dropped as blank (blank_score %.2f < threshold %g; --blank-threshold N changes it, --keep-blank keeps blanks)",
                         path, r.stats.get("page", r.index + 1), r.stats.get("blank_score") or 0.0, opts.blank_threshold)
        if opts.drop_blank and all(r.encoded is None for r in results):
            log.warning("%s: every page looks blank (blank_score < %g); keeping all pages", path, opts.blank_threshold)
            results = run_pages(pdfio.iter_page_inputs(pdf, name=path.name), n, opts, allow_drop=False, plan=plan, fetch=fetch, hints=hints)
            for r in results:
                r.stats["blank"] = True
                if (t := t_maps.get(r.index)) is not None:
                    r.stats["t"]["duplex_maps"] = t
    kept = [r for r in results if r.encoded is not None]
    if any(r.stats.get("encoding", {}).get("mask_codec") == "g4" for r in kept):
        log.warning("%s: jbig2 unavailable or failed; ink masks use the CCITT G4 fallback (larger files)", path)
    meta = pdfio.build_metadata(src_info, target.stem)
    tmp = safety.make_temp_path(target)
    try:
        pdfio.write_pdf([r.encoded for r in kept], tmp, meta)
        pdfio.validate_pdf(tmp, len(kept))
        final = _install(tmp, target, path, fingerprint, res)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    res.pages_out = len(kept)
    if final is not None:
        res.output = str(final)
        res.bytes_out = final.stat().st_size
    res.pages = [r.stats for r in results]
    res.stage_cpu = _sum_stages(res.pages)
    res.wall = time.perf_counter() - t_wall
    return res


def process_image_file(path: Path, output: Path | None, opts: Options, keep_original: str | None = None,
                       source: Path | None = None, backup_done: bool = False,
                       fingerprint: dict | None = None) -> FileResult:
    path = Path(path)
    src = Path(source) if source else path
    target = Path(output) if output else path
    t_wall = time.perf_counter()
    res = FileResult(str(path), "processed", bytes_in=src.stat().st_size, pages_in=1, pages_out=1)
    t0 = time.process_time()
    rgb, dpi, fmt = imageio.read_image(src)
    t_dec = time.process_time() - t0
    if not backup_done:
        res.backup = str(p) if (p := _prepare_target(path, output, opts, keep_original)) else None
    if opts.ocr:
        log.info("%s: --ocr does not apply to image files (no place for a text layer); ignored", path)
    out, out_dpi, stats = process_array(rgb, dpi, opts, allow_drop=False)
    t0 = time.process_time()
    data = imageio.encode_image(out, out_dpi, fmt)
    stats["t"]["decode"] = t_dec
    stats["t"]["encode"] = time.process_time() - t0
    stats["page"] = 1
    _log_folds(path.name, stats)
    stats["bytes"] = len(data)
    tmp = safety.make_temp_path(target)
    try:
        tmp.write_bytes(data)
        imageio.validate_image(tmp, (out.shape[1], out.shape[0]), fmt)
        final = _install(tmp, target, path, fingerprint, res)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if final is not None:
        res.output = str(final)
        res.bytes_out = final.stat().st_size
    res.pages = [stats]
    res.stage_cpu = _sum_stages(res.pages)
    res.wall = time.perf_counter() - t_wall
    return res


def process_file(path, opts: Options, output=None, keep_original: str | None = None, source=None,
                 backup_done: bool = False, fingerprint: dict | None = None) -> FileResult:
    """Process one file in place (or to `output`). Never modifies the input unless it is backed up first
    (or --no-backup), and replaces it atomically only after the new file validated.
    Queued jobs pass `source` (the backup copy to read), `backup_done` and the enqueue-time `fingerprint`."""
    path = Path(path)
    fmt = imageio.sniff_format(Path(source) if source else path)
    out = Path(output) if output else None
    kw = dict(source=source, backup_done=backup_done, fingerprint=fingerprint)
    if fmt == "PDF":
        return process_pdf_file(path, out, opts, keep_original, **kw)
    if fmt in imageio.SUPPORTED:
        return process_image_file(path, out, opts, keep_original, **kw)
    log.info("%s: unsupported file type; leaving untouched", path)
    return FileResult(str(path), "skipped", "unsupported file type", bytes_in=path.stat().st_size)


def write_report(path, opts: Options, results: list) -> None:
    import json
    from . import __version__
    Path(path).write_text(json.dumps(
        {"version": __version__, "duplex_version": duplex.DUPLEX_VERSION, "fold_version": fold.FOLD_VERSION, "options": asdict(opts), "files": [asdict(r) for r in results]}, indent=1))
