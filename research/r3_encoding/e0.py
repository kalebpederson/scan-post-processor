"""E0 baseline: user's current pipeline run on the ORIGINAL (uncleaned) page, wrapped as a PDF first (lossless embed via img2pdf)."""
import subprocess, time, os, uuid, shutil
import img2pdf, cv2
from config import *

CMD = ["magick", "-density", "300", None, "-level", "5.098%,86.6667%,0.8822", "-density", "200", "-page", "Letter", "-quality", "70", None]

def original_pdf(p, out):
    src = p["file"]
    if p["kind"] == "epson":
        rot = os.path.join(TMP, p["name"] + "_rot.png"); cv2.imwrite(rot, cv2.cvtColor(load_src(p), cv2.COLOR_RGB2BGR)); src = rot
    layout = img2pdf.get_fixed_dpi_layout_fun((p["native"], p["native"]))
    open(out, "wb").write(img2pdf.convert(src, layout_fun=layout))

def run_e0(p, outdir):
    inp = os.path.join(outdir, p["name"] + "_orig.pdf"); out = os.path.join(outdir, p["name"] + ".pdf")
    original_pdf(p, inp)
    cmd = list(CMD); cmd[3] = inp; cmd[-1] = out
    t = time.perf_counter(); subprocess.run(cmd, check=True, capture_output=True); dt = time.perf_counter() - t
    return out, dt
