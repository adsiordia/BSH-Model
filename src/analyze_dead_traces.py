"""Enzymes with no active product: is there ANY trace signal, and on which amines?"""
from pathlib import Path
import sys, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
pd.set_option("display.width", 200)

lab = lb.build(min_reps=1)
lab["max_intensity"] = lab.max_intensity.fillna(0.0)
per = lab.groupby("Enzyme").active.sum()
dead = sorted(per[per == 0].index)
print(f"{lab.Enzyme.nunique()} enzymes; {len(dead)} make NO active product at 50,000\n")
print("dead enzymes:", ", ".join(dead), "\n")

d = lab[lab.Enzyme.isin(dead)]
nz = d[d.max_intensity > 0].sort_values(["Enzyme", "max_intensity"], ascending=[True, False])
print(f"=== every non-zero reading across those {len(dead)} enzymes "
      f"({len(nz)} cells of {len(d):,}) ===")
print(nz[["Enzyme","Amine","Hydroxyl","max_intensity","n_reps_above","n_features"]].to_string(
      index=False, float_format=lambda v: f"{v:,.0f}"))

print("\n=== which amines show any trace at all in the dead enzymes ===")
t = nz.groupby("Amine").agg(cells=("Amine","size"), best=("max_intensity","max"),
                            enzymes=("Enzyme","nunique")).sort_values("best", ascending=False)
print(t.to_string(float_format=lambda v: f"{v:,.0f}"))

print("\n=== per enzyme ===")
for e in dead:
    s = nz[nz.Enzyme == e]
    if len(s):
        bits = ", ".join(f"{r.Amine}+{r.Hydroxyl} {r.max_intensity:,.0f}" for r in s.itertuples())
        print(f"  {e:12s} {len(s)} trace(s): {bits}")
    else:
        print(f"  {e:12s} nothing at all - every one of its 75 cells read exactly zero")

print("\n=== for scale, what an ACTIVE enzyme looks like ===")
live = per[per > 0]
print(f"  {len(live)} active enzymes, median {live.median():.0f} products each "
      f"(range {live.min():.0f}-{live.max():.0f})")
la = lab[lab.Enzyme.isin(live.index) & lab.active]
print(f"  their active cells: median intensity {la.max_intensity.median():,.0f}")
print(f"  highest trace in any dead enzyme: {nz.max_intensity.max():,.0f}  "
      f"(the 50,000 line is {lb.THRESHOLD:,.0f})")
