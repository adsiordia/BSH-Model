"""Test the collaborators' hypothesis directly.

Their claim: E. coli signal peptidases handle Bacteroidota signal peptides better
than Bacillota ones, so BACILLOTA sequences (especially long peptides) should be
the ones showing no activity in the assay.

That makes three checkable predictions:
  P1  inactive enzymes should be enriched for Bacillota
  P2  inactive enzymes should carry longer signal peptides
  P3  within Bacillota, longer peptide -> more likely inactive
"""
from pathlib import Path
import sys, json, numpy as np, pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, pointbiserialr
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
pd.set_option("display.width", 210)

lab = lb.build(min_reps=1); truth = lab.groupby("Enzyme").active.sum()
P = pd.DataFrame(json.load(open(ROOT/"site/candidates.json"))["proteins"])
FILL = {"Parabacteroides":"Bacteroidota","Paraprevotella":"Bacteroidota","Odoribacter":"Bacteroidota",
        "Barnesiella":"Bacteroidota","Prevotella":"Bacteroidota","Dysgonomonas":"Bacteroidota",
        "Butyricimonas":"Bacteroidota","Muribaculaceae":"Bacteroidota","Alistipes":"Bacteroidota",
        "Homeothermus":"Bacteroidota","Okeania":"Cyanobacteriota","Symploca":"Cyanobacteriota",
        "Moorea":"Cyanobacteriota","Obscuribacterales":"Cyanobacteriota"}
def fill(r):
    if r.phylum: return r.phylum
    for t in (r.organism or "").replace("Candidatus ","").split()[:2]:
        if t.strip("[]") in FILL: return FILL[t.strip("[]")]
    return "other"
P["phy"] = P.apply(fill, axis=1)
P["n_active"] = P.accession.map(truth)
m = P[P.n_active.notna()].copy()
m["dead"] = m.n_active == 0
print(f"{len(m)} enzymes with a signal peptide AND an assay result "
      f"(every one of the {len(P)} carries a predicted signal peptide)\n")

print("=== P1: are the inactive enzymes Bacillota? ===")
t = m.groupby("phy").agg(enzymes=("accession","size"), inactive=("dead","sum")).sort_values("enzymes", ascending=False)
t["rate"] = (t.inactive/t.enzymes).map(lambda v: f"{v:.0%}")
print(t.to_string())
a = int(m[m.phy=="Bacillota"].dead.sum()); na = int((m.phy=="Bacillota").sum())
b = int(m[m.phy=="Bacteroidota"].dead.sum()); nb = int((m.phy=="Bacteroidota").sum())
_, p = fisher_exact([[a, na-a],[b, nb-b]])
print(f"\n  Bacillota    {a}/{na} inactive")
print(f"  Bacteroidota {b}/{nb} inactive")
print(f"  Fisher exact p = {p:.4f}")
print(f"  -> prediction P1 is {'SUPPORTED' if a/na > b/nb else 'CONTRADICTED'}: "
      f"the failures are {'Bacillota' if a/na > b/nb else 'Bacteroidota'}")

print("\n=== P2: do inactive enzymes have longer signal peptides? ===")
d, l = m[m.dead].cut.dropna(), m[~m.dead].cut.dropna()
_, p2 = mannwhitneyu(d, l)
print(f"  inactive: median {d.median():.0f} aa (range {d.min():.0f}-{d.max():.0f}, n={len(d)})")
print(f"  active  : median {l.median():.0f} aa (range {l.min():.0f}-{l.max():.0f}, n={len(l)})")
print(f"  Mann-Whitney p = {p2:.3f}  -> {'SUPPORTED' if p2<0.05 and d.median()>l.median() else 'NOT SUPPORTED'}")
r, pr = pointbiserialr(m.dead.astype(int), m.cut.fillna(m.cut.median()))
print(f"  correlation between peptide length and being inactive: r = {r:+.3f}, p = {pr:.3f}")

print("\n=== the longest signal peptides in the set -- are they the dead ones? ===")
top = m.nlargest(12, "cut").copy()
top["short"] = top.organism.str.split().str[:2].str.join(" ")
top["result"] = np.where(top.dead, "MADE NOTHING", top.n_active.astype(int).astype(str)+" products")
print(top[["accession","short","phy","cut","result"]].to_string(index=False))

print("\n=== P3: signal peptide length by phylum ===")
for ph in ("Bacillota","Bacteroidota"):
    s = m[m.phy==ph].cut.dropna()
    print(f"  {ph:14s} n={len(s):2d}  median {s.median():.0f} aa  "
          f"range {s.min():.0f}-{s.max():.0f}  mean {s.mean():.1f}")
x, y = m[m.phy=="Bacillota"].cut.dropna(), m[m.phy=="Bacteroidota"].cut.dropna()
_, p3 = mannwhitneyu(x, y)
print(f"  Bacillota vs Bacteroidota peptide length: Mann-Whitney p = {p3:.3f}")
print(f"  -> Bacillota peptides are {'LONGER' if x.median()>y.median() else 'SHORTER OR EQUAL'} "
      f"than Bacteroidota ones")
