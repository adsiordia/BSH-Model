"""What amines were these trimmed enzymes predicted to act on, and was there
ever ANY intensity recorded in those cells?"""
from pathlib import Path
import sys, pandas as pd, numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
OUT = ROOT/"outputs"
pd.set_option("display.width", 220)

lab = lb.build(min_reps=1)
print("measurement table columns:", list(lab.columns), "\n")

calls = pd.read_csv(OUT/"gained_calls.csv")
keep = ["Enzyme","Amine","Hydroxyl","max_intensity","n_reps_above","detected","active"]
keep = [c for c in keep if c in lab.columns]
j = calls.merge(lab[keep], left_on=["accession","Amine","Hydroxyl"],
                right_on=["Enzyme","Amine","Hydroxyl"], how="left")
j["max_intensity"] = j.max_intensity.fillna(0.0)
print(f"{len(j)} new calls, matched to the measurement table: "
      f"{int(calls.shape[0] - j.max_intensity.isna().sum())}\n")

print("=== every one of the new calls, with what was actually recorded ===")
show = j[["accession","Amine","Hydroxyl","raw_full","raw_trimmed",
          "max_intensity","n_reps_above","detected"]].sort_values(
          ["accession","raw_trimmed"], ascending=[True,False])
show = show.rename(columns={"raw_full":"pred_full","raw_trimmed":"pred_trim",
                            "max_intensity":"intensity","n_reps_above":"reps_above_50k"})
print(show.to_string(index=False, float_format=lambda v: f"{v:,.4f}" if v < 100 else f"{v:,.0f}"))

print("\n=== was there EVER any intensity in these cells? ===")
nz = j.max_intensity > 0
print(f"  cells with a non-zero reading in any replicate : {int(nz.sum())} of {len(j)}")
print(f"  cells that read exactly zero                   : {int((~nz).sum())} of {len(j)}")
if nz.any():
    print(f"  among non-zero cells: min {j.max_intensity[nz].min():,.0f}  "
          f"median {j.max_intensity[nz].median():,.0f}  max {j.max_intensity[nz].max():,.0f}")
    for lo, hi, t in [(1,10_000,"under 10,000 (below even the old cut-off)"),
                      (10_000,25_000,"10,000 - 25,000"),
                      (25_000,50_000,"25,000 - 50,000 (active under the OLD rule)"),
                      (50_000,np.inf,"above 50,000")]:
        print(f"    {t:46s} {int(((j.max_intensity>=lo)&(j.max_intensity<hi)).sum()):3d}")

print("\n=== which amines were predicted ===")
t = j.groupby("Amine").agg(calls=("Amine","size"),
                           any_signal=("max_intensity", lambda s: int((s>0).sum())),
                           best_intensity=("max_intensity","max"),
                           mean_pred=("raw_trimmed","mean")).sort_values("calls", ascending=False)
print(t.to_string(float_format=lambda v: f"{v:,.3f}" if v < 100 else f"{v:,.0f}"))

print("\n=== the 8 proteins: anything detected anywhere in their 75 cells? ===")
for acc in sorted(j.accession.unique()):
    cells = lab[lab.Enzyme == acc]
    mi = cells.max_intensity.fillna(0) if "max_intensity" in cells else pd.Series([0])
    print(f"  {acc:12s} {len(cells):2d} cells | non-zero {int((mi>0).sum()):2d} | "
          f"best {mi.max():>12,.0f} | active at 50k {int(cells.active.sum()) if 'active' in cells else 0}")
