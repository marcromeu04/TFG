"""ROIs (ppm_lo, ppm_hi) for the 12 OCM metabolites, anchored to HMDB peaks at serum pH ~7.4.
28 ROIs total; after water (4.65-4.95 ppm) exclusion, 27 active ROIs * 3 stats = 81 features."""
from __future__ import annotations

# (metabolite, ppm_lo, ppm_hi, comment)
# Built from HMDB peak listings + Chenomx default profiling regions.
ROIS: tuple[tuple[str, float, float, str], ...] = (
    # Lactate
    ("Lactate",     1.30, 1.36, "doublet at 1.33 ppm (CH3)"),
    ("Lactate",     4.08, 4.14, "quartet at 4.11 ppm (CH); near water"),
    # L-Histidine
    ("L-Histidine", 7.04, 7.12, "imidazole H at 7.08 ppm"),
    ("L-Histidine", 7.85, 7.92, "imidazole H at 7.88 ppm"),
    ("L-Histidine", 3.10, 3.20, "beta-CH2 at 3.15 ppm"),
    # L-Cysteine
    ("L-Cysteine",  3.00, 3.08, "beta-CH2 at 3.04 ppm"),
    ("L-Cysteine",  3.95, 4.05, "alpha-CH at 4.00 ppm; near water"),
    # D-Glucose (alpha + beta anomers)
    ("D-Glucose",   3.20, 3.30, "H2 of alpha/beta anomers"),
    ("D-Glucose",   3.40, 3.55, "H3, H4, H5 region"),
    ("D-Glucose",   5.20, 5.27, "anomeric H of alpha-Glucose"),
    # L-Glycine
    ("L-Glycine",   3.55, 3.62, "singlet at 3.56 ppm (CH2)"),
    # Betaine
    ("Betaine",     3.24, 3.30, "singlet at 3.26 ppm (N(CH3)3)"),
    ("Betaine",     3.88, 3.96, "singlet at 3.92 ppm (CH2)"),
    # Pyruvate
    ("Pyruvate",    2.34, 2.40, "singlet at 2.37 ppm (CH3)"),
    # L-Threonine
    ("L-Threonine", 1.30, 1.36, "doublet at 1.33 ppm (CH3); overlaps Lactate"),
    ("L-Threonine", 3.55, 3.62, "alpha-CH near 3.58 ppm"),
    ("L-Threonine", 4.20, 4.30, "beta-CH at 4.25 ppm"),
    # L-Serine
    ("L-Serine",    3.83, 3.90, "alpha-CH at 3.85 ppm"),
    ("L-Serine",    3.94, 4.02, "beta-CH2 at 3.98 ppm"),
    # Choline
    ("Choline",     3.18, 3.22, "singlet at 3.20 ppm (N(CH3)3)"),
    ("Choline",     3.50, 3.55, "alpha-CH2 at 3.51 ppm"),
    ("Choline",     4.04, 4.10, "beta-CH2 at 4.07 ppm; near water"),
    # Creatine
    ("Creatine",    3.02, 3.06, "singlet at 3.04 ppm (N-CH3)"),
    ("Creatine",    3.92, 3.96, "singlet at 3.93 ppm (CH2); near water"),
    # Creatinine
    ("Creatinine",  3.04, 3.08, "singlet at 3.05 ppm (N-CH3)"),
    ("Creatinine",  4.04, 4.08, "singlet at 4.06 ppm (CH2); near water"),
)


def active_rois(water_min: float, water_max: float):
    """Return ROIs whose midpoint falls outside the water region."""
    out = []
    for met, lo, hi, comment in ROIS:
        mid = 0.5 * (lo + hi)
        if water_min <= mid <= water_max:
            continue
        out.append((met, lo, hi, comment))
    return tuple(out)
