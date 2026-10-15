#!/bin/bash
# Full study. Change cleaner:  CLEANER_DIR=/abs/path/to/dir_with_clean.py ./run_all.sh   (dir must expose clean.clean_page(rgb, dpi, paper="flatten"))
# Results land in work/<basename of CLEANER_DIR>/. Default (no env) = clean_snapshot/ ; the reported numbers use clean_final/.
set -e
cd "$(dirname "$0")"; export PYTHONWARNINGS=ignore
PY=${PY:-python3}
$PY prep.py                       # clean + classify (blank / gray / colour)
$PY run_study.py '*'              # encode -> PDF -> poppler render -> fidelity -> OCR, every candidate x page
$PY run_e0.py                     # E0 baseline (magick pipeline)
$PY run_pdfium_ocr.py 'E3_c75*,E3_base,E1_j150_q40_420,E1_j150_q55_420,E1_j200_*_420,E4_s2,E4_s4,E4_s7,E2_p150_k8_km,E2_p200_k16_km'
$PY epson_clean_vs_raw.py         # cleaned vs uncleaned Epson legibility
$PY haze.py                       # bytes cost of faint haze
$PY compat.py E1_j150_q40_420,E2_p150_k8_km,E3_c75_q40,E3_c75_q40_hl25,E3_base,E3_g4,E3_m200,E3_m400,E4_s7,E4_s2,E4_s7_g4,E3g_c75_q40,E1g_j150_q55 single13,pair09_front,pair01_front,pair07_front,single26 300
$PY timing.py
$PY run_metrics2.py "E1_j*,E2_p*_km,E3_*,E3r_*,E3x_*,E4_*,E5*,E0,E0_noalpha"   # round-2 metrics
$PY aggregate2.py
$PY make_report.py                # tables.md + pareto_*.png
$PY make_samples.py
# panels: see panels.py usage; commands used are listed in REPORT.md
