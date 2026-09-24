"""Enzymes with signal peptides make fewer products. Is that reduced YIELD
(consistent with partial processing) or genuinely narrower specificity?

Reduced yield  -> their products are systematically weaker, and the gap shrinks
                  as the intensity threshold is lowered.
Narrower range -> the gap persists at any threshold and intensities look normal.
"""
from pathlib import Path
import sys, h5py, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
pd.set_option("display.width", 200)

with h5py.File(ROOT/"data/embeddings/proteins_without_signal_peptides_trimmed.h5") as f:
    SP = {k.split("_")[-1] for k in f.keys()}

print("=== does the gap close as the threshold drops? ===")
hdr = f"  {'threshold':>10s} {'with SP':>9s} {'without':>9s} {'ratio':>7s}   {'p':>8s}"
print(hdr); print("  " + "-"*(len(hdr)-2))
rows = []
for th in (5_000, 10_000, 25_000, 50_000, 100_000, 250_000):
    a = lb.build(threshold=th, min_reps=1)
    n = a.groupby("Enzyme").active.sum()
    w = n[[e in SP for e in n.index]]
    o = n[[e not in SP for e in n.index]]
    wa, oa = w[w > 0], o[o > 0]          # enzymes that work at this threshold
    _, p = mannwhitneyu(wa, oa)
    rows.append((th, wa.median(), oa.median(), wa.median()/oa.median(), p))
    print(f"  {th:10,d} {wa.median():9.0f} {oa.median():9.0f} "
          f"{wa.median()/oa.median():7.2f}   {p:8.5f}")

print("\n=== are their detected products weaker in intensity? ===")
a = lb.build(min_reps=1)
a["sp"] = a.Enzyme.isin(SP)
det = a[a.max_intensity.fillna(0) > 0]
for lbl, s in [("with a signal peptide", det[det.sp]), ("without", det[~det.sp])]:
    print(f"  {lbl:24s} {len(s):5,d} detected cells, median intensity "
          f"{s.max_intensity.median():12,.0f}")
_, p = mannwhitneyu(det[det.sp].max_intensity, det[~det.sp].max_intensity)
print(f"  Mann-Whitney p = {p:.2e}")

print("\n=== among ACTIVE cells only (already above 50,000) ===")
act = a[a.active]
for lbl, s in [("with a signal peptide", act[act.sp]), ("without", act[~act.sp])]:
    print(f"  {lbl:24s} {len(s):5,d} active cells, median intensity "
          f"{s.max_intensity.median():12,.0f}")
_, p2 = mannwhitneyu(act[act.sp].max_intensity, act[~act.sp].max_intensity)
print(f"  Mann-Whitney p = {p2:.2e}")

print("\n=== how many replicates do their detections survive? ===")
for lbl, s in [("with a signal peptide", det[det.sp]), ("without", det[~det.sp])]:
    print(f"  {lbl:24s} mean reps above cut-off {s.n_reps_above.mean():.2f}")
