| metric | B0 | FINAL_flatten | FINAL_keep | M1_flatfield_levels | M2_threshold_matched | M2_threshold_retain |
|---|---|---|---|---|---|---|
| paper whiteness, 53 white-paper pages (mean / worst page) | 1.0000 / 0.9999 | 1.0000 / 0.9999 | 1.0000 / 0.9999 | 1.0000 / 0.9995 | 1.0000 / 1.0000 | 1.0000 / 1.0000 |
| background-clean fraction (non-ink area >=250) | 0.966 | 0.996 | 0.967 | 0.998 | 0.999 | 0.996 |
| colored paper single08: whiteness (flatten) | 0.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 |
| colored paper single38: whiteness (flatten) | 0.000 | 0.995 | 0.000 | 0.999 | 0.999 | 0.997 |
| colored paper image01: whiteness (flatten) | 0.847 | 0.992 | 0.992 | 0.999 | 0.998 | 0.995 |
| ink median / p90 luminance (0=black) | 60 / 75 | 38 / 53 | 37 / 53 | 54 / 70 | 48 / 62 | 48 / 62 |
| ink-area ratio (dark px out/in) | 0.98 | 1.30 | 1.33 | 0.97 | 1.15 | 1.15 |
| hue shift mean / p90 (deg) | 3.4 / 8.1 | 2.2 / 4.7 | 2.2 / 4.7 | 2.6 / 5.8 | 1.6 / 3.3 | 1.6 / 3.3 |
| chroma ratio median / p10 | 1.24 / 0.94 | 1.20 / 1.15 | 1.20 / 1.15 | 1.25 / 1.00 | 1.14 / 1.11 | 1.14 / 1.11 |
| faint-content retention (17 regions, core px) | 0.663 | 0.851 | 0.895 | 0.671 | 0.727 | 0.791 |
|   - pencil | 0.89 | 1.00 | 1.00 | 0.93 | 1.00 | 1.00 |
|   - grid | 0.47 | 0.60 | 0.75 | 0.34 | 0.34 | 0.48 |
|   - light_print | 0.47 | 0.83 | 0.83 | 0.51 | 0.57 | 0.69 |
|   - ruled | 0.75 | 1.00 | 1.00 | 0.79 | 0.99 | 0.99 |
|   - highlighter | 0.74 | 1.00 | 1.00 | 0.90 | 0.99 | 1.00 |
|   - light_pen | 0.97 | 1.00 | 1.00 | 0.98 | 1.00 | 1.00 |
| faint contrast: out dE / native dE | 19.3 / 24.5 | 31.5 / 24.5 | 32.8 / 24.5 | 19.6 / 24.5 | 24.1 / 24.5 | 25.4 / 24.5 |
| bleed-only regions survival (11 regions) | 0.098 | 0.016 | 0.107 | 0.029 | 0.020 | 0.205 |
| pair bleed survival, pixel-pooled (35 directed) | 0.319 | 0.039 | 0.352 | 0.016 | 0.007 | 0.091 |
| pair bleed survival, per-instance mean | 0.069 | 0.080 | 0.137 | 0.031 | 0.019 | 0.154 |
| mean darkness in bleed region (native visible-bleed darkness 0.0515) | 0.0068 | 0.0112 | 0.0216 | 0.0016 | 0.0019 | 0.0117 |