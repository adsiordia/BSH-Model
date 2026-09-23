"""For each predicted (amine, core) product: is that product ever active
ANYWHERE in the dataset, across all enzymes?"""
from pathlib import Path
import sys, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
OUT = ROOT/"outputs"; pd.set_option("display.width", 200)

lab = lb.build(min_reps=1)
n_enz = lab.Enzyme.nunique()
print(f"whole dataset: {n_enz} enzymes, {len(lab):,} cells, "
      f"{int(lab.active.sum()):,} active ({lab.active.mean():.1%})\n")

# how often is each (amine, core) product made, across every enzyme?
prod = lab.groupby(["Amine","Hydroxyl"]).agg(
    enzymes_active=("active","sum"),
    enzymes_tested=("active","size"),
    any_signal=("max_intensity", lambda s: int((s.fillna(0)>0).sum())),
    best=("max_intensity","max")).reset_index()
prod["pct_active"] = prod.enzymes_active / prod.enzymes_tested

calls = pd.read_csv(OUT/"gained_calls.csv")
pairs = calls.groupby(["Amine","Hydroxyl"]).size().reset_index(name="n_calls")
j = pairs.merge(prod, on=["Amine","Hydroxyl"], how="left").sort_values(
    "enzymes_active", ascending=False)

print("=== the products the trimmed model predicted, and how often they are made ===")
print(j[["Amine","Hydroxyl","n_calls","enzymes_active","enzymes_tested",
         "pct_active","best"]].to_string(
      index=False, float_format=lambda v: f"{v:,.1%}" if v <= 1 else f"{v:,.0f}"))

print(f"\n  distinct products predicted: {len(j)}")
print(f"  of those, ever active in at least one enzyme: "
      f"{int((j.enzymes_active>0).sum())} of {len(j)}")
print(f"  never active in any enzyme: {int((j.enzymes_active==0).sum())}")

print("\n=== how do these compare with the panel as a whole? ===")
print(f"  all {len(prod)} products      : median {prod.pct_active.median():.1%} of enzymes active")
print(f"  the {len(j)} predicted ones   : median {j.pct_active.median():.1%} of enzymes active")
top = prod.sort_values("pct_active", ascending=False).head(12)
print("\n  the most commonly made products in the whole dataset:")
for r in top.itertuples():
    mark = "  <- predicted" if ((j.Amine==r.Amine)&(j.Hydroxyl==r.Hydroxyl)).any() else ""
    print(f"    {r.Amine:28s} {r.Hydroxyl:5s} {r.pct_active:6.1%}{mark}")
