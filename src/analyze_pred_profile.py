"""What the trimmed-sequence model predicts these proteins will make."""
from pathlib import Path
import sys, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
OUT = ROOT/"outputs"; CUT = 0.5; pd.set_option("display.width", 200)

g = pd.read_csv(OUT/"candidate_predictions.csv")
lab = lb.build(min_reps=1)
meas = lab.groupby(["Amine","Hydroxyl"]).active.mean().rename("measured_rate")

print(f"{g.entry.nunique()} proteins, {len(g):,} cells, "
      f"{int((g.raw_trimmed>=CUT).sum())} predicted active (trimmed)\n")

hit = g[g.raw_trimmed >= CUT]

print("=== which amines are they predicted to act on ===")
a = hit.groupby("Amine").agg(calls=("Amine","size"), proteins=("entry","nunique"),
                             mean_conf=("raw_trimmed","mean")).sort_values("calls", ascending=False)
a["pct_of_proteins"] = a.proteins / g.entry.nunique()
print(a.to_string(float_format=lambda v: f"{v:,.3f}"))

print("\n=== which exact products (amine + core) ===")
p = hit.groupby(["Amine","Hydroxyl"]).agg(calls=("Amine","size"),
                                          mean_conf=("raw_trimmed","mean")).reset_index()
p = p.merge(meas.reset_index(), on=["Amine","Hydroxyl"], how="left")
p["pred_rate"] = p.calls / g.entry.nunique()
p = p.sort_values("calls", ascending=False)
print(p[["Amine","Hydroxyl","calls","pred_rate","measured_rate","mean_conf"]].to_string(
      index=False, float_format=lambda v: f"{v:,.3f}"))

print("\n=== by bile acid core ===")
print(hit.groupby("Hydroxyl").size().rename("calls").to_string())

print("\n=== the 8 proteins that were NEVER measured - the actual new leads ===")
novel = g[g.measured.isna()]
for acc, sub in novel.groupby("accession"):
    h = sub[sub.raw_trimmed >= CUT].sort_values("raw_trimmed", ascending=False)
    print(f"\n  {acc}  -  {len(h)} products predicted")
    for r in h.itertuples():
        print(f"      {r.Amine:28s} {r.Hydroxyl:5s}  {r.raw_trimmed:.3f}"
              f"   (full sequence: {r.raw_full:.3f})")

print("\n=== amines predicted for the never-measured proteins only ===")
nh = novel[novel.raw_trimmed >= CUT]
print(nh.groupby("Amine").agg(calls=("Amine","size"),
                              proteins=("accession","nunique")).sort_values(
      "calls", ascending=False).to_string())
