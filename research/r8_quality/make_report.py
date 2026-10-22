"""Assemble REPORT.md from REPORT.md.in and the aggregate outputs in work/ (run the study/aggregate scripts first)."""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
w = here / "work"
py = sys.executable
levels = subprocess.run([py, str(here / "summarize_levels.py")], capture_output=True, text=True, check=True).stdout
timing = subprocess.run([py, str(here / "summarize_timing.py")], capture_output=True, text=True, check=True).stdout
s = (here / "REPORT.md.in").read_text()
s = s.replace("@@LADDER@@", (w / "ladder.md").read_text().strip()).replace("@@SWEEP@@", (w / "sweep.md").read_text().strip())
s = s.replace("@@SWEEP2@@", (w / "sweep2.md").read_text().strip()).replace("@@LEVELS@@", levels.strip()).replace("@@TIMING@@", timing.strip())
(here / "REPORT.md").write_text(s)
