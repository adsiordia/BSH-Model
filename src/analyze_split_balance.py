"""Is the dev/test split balanced, or does it skew amine preference?"""
from pathlib import Path
import sys, numpy as np, pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
pd.set_option("display.width", 210)

lab = lb.build(min_reps=1)
sp = pd.read_csv(ROOT/"data/derived/holdout_split.csv")
lab = lab.merge(sp[["Enzyme","split","cluster"]], on="Enzyme", how="left")
print("=== size and overall rate ===")
t = lab.groupby("split").agg(enzymes=("Enzyme","nunique"), clusters=("cluster","nunique"),
                             cells=("active","size"), active=("active","sum"))
t["rate"] = (t.active/t.cells)
print(t.to_string(float_format=lambda v: f"{v:,.4f}"))
d, e = lab[lab.split=="dev"], lab[lab.split=="test"]
_, p = fisher_exact([[int(d.active.sum()), int((~d.active).sum())],
                     [int(e.active.sum()), int((~e.active).sum())]])
print(f"\n  dev {d.active.mean():.4f} vs test {e.active.mean():.4f}   Fisher p = {p:.4f}")

print("\n=== breadth: are test enzymes broader or narrower? ===")
b = lab.groupby(["Enzyme","split"]).active.sum().reset_index()
for s in ("dev","test"):
    x = b[b.split==s].active
    print(f"  {s:5s} n={len(x):3d}  median {x.median():5.1f}  mean {x.mean():5.1f}  "
          f"range {x.min():.0f}-{x.max():.0f}  silent {int((x==0).sum())}")
_, pb = mannwhitneyu(b[b.split=="dev"].active, b[b.split=="test"].active)
print(f"  Mann-Whitney p = {pb:.3f}")

print("\n=== per-amine activity rate, dev vs test ===")
r = lab.pivot_table(index="Amine", columns="split", values="active", aggfunc="mean")
n = lab.pivot_table(index="Amine", columns="split", values="active", aggfunc="sum")
r["diff"] = r["test"] - r["dev"]
r["n_dev"], r["n_test"] = n["dev"].astype(int), n["test"].astype(int)
r = r.sort_values("diff")
print((r.assign(dev=(r["dev"]*100).round(1), test=(r["test"]*100).round(1),
                diff=(r["diff"]*100).round(1))[["dev","test","diff","n_dev","n_test"]]).to_string())

print("\n=== amines whose rate differs most (Fisher per amine) ===")
rows = []
for a, s in lab.groupby("Amine"):
    dd, tt = s[s.split=="dev"], s[s.split=="test"]
    if not len(tt): continue
    tab = [[int(dd.active.sum()), int((~dd.active).sum())],
           [int(tt.active.sum()), int((~tt.active).sum())]]
    _, pv = fisher_exact(tab)
    rows.append((a, dd.active.mean(), tt.active.mean(), tt.active.mean()-dd.active.mean(), pv))
R = pd.DataFrame(rows, columns=["Amine","dev","test","diff","p"]).sort_values("p")
R["bonferroni"] = (R.p*len(R)).clip(upper=1)
print(R.head(8).to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
print(f"\n  amines with raw p<0.05: {int((R.p<0.05).sum())} of {len(R)}")
print(f"  surviving Bonferroni   : {int((R.bonferroni<0.05).sum())}")

print("\n=== per-core rate ===")
c = lab.pivot_table(index="Hydroxyl", columns="split", values="active", aggfunc="mean")
print((c*100).round(1).to_string())
