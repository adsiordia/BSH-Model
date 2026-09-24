"""Are gut Bacteroidetes BSH homologs disproportionately inactive in this assay?

Limited to the 58 enzymes that were both measured AND carry organism metadata
(the signal-peptide set). Organism names are not available for all 115.
"""
from pathlib import Path
import sys, json, numpy as np, pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
pd.set_option("display.width", 200)

lab = lb.build(min_reps=1)
truth = lab.groupby("Enzyme").active.sum()
P = pd.DataFrame(json.load(open(ROOT/"site/candidates.json"))["proteins"])
P["n_active"] = P.accession.map(truth)
m = P[P.n_active.notna()].copy()
m["dead"] = m.n_active == 0
GUT = ("Bacteroides", "Parabacteroides", "Alistipes")
m["gut"] = m.organism.fillna("").str.startswith(GUT)
print(f"{len(m)} measured enzymes with organism names "
      f"(of {lab.Enzyme.nunique()} measured overall)\n")

print("=== made nothing at all, by group ===")
for lbl, s in [("Bacteroides / Parabacteroides / Alistipes", m[m.gut]),
               ("every other organism", m[~m.gut])]:
    print(f"  {lbl:42s} {len(s):2d} enzymes, {int(s.dead.sum())} inactive "
          f"({s.dead.mean():5.1%})")
tab = [[int(m[m.gut].dead.sum()), int((~m[m.gut].dead).sum())],
       [int(m[~m.gut].dead.sum()), int((~m[~m.gut].dead).sum())]]
_, p = fisher_exact(tab)
print(f"\n  Fisher exact p = {p:.5f}")

print("\n=== every enzyme from those three genera ===")
g = m[m.gut].copy()
g["short"] = g.organism.str.split().str[:2].str.join(" ")
g["flag"] = np.where(g.dead, "<- made NOTHING", "")
print(g.sort_values("n_active")[["accession","short","n_active","flag"]].to_string(index=False))

print("\n=== the inactive enzymes, all of them ===")
d = m[m.dead].copy(); d["short"] = d.organism.str.split().str[:2].str.join(" ")
print(d[["accession","short"]].to_string(index=False))
print(f"\n  {int(d.gut.sum())} of {len(d)} are from those three genera")

print("\n=== when they do work, are they narrower? ===")
a, b = m[m.gut & ~m.dead].n_active, m[~m.gut & ~m.dead].n_active
_, pw = mannwhitneyu(a, b)
print(f"  active gut Bacteroidetes: median {a.median():.0f} products (n={len(a)})")
print(f"  active others           : median {b.median():.0f} products (n={len(b)})")
print(f"  Mann-Whitney p = {pw:.3f}")

print("\n=== is it the genus, or just Bacteroidota broadly? ===")
m["phy_bact"] = m.phylum.fillna("").eq("Bacteroidota") | m.gut
for lbl, s in [("Bacteroidota (incl. marine/environmental)", m[m.phy_bact]),
               ("the three gut genera only", m[m.gut]),
               ("Bacteroidota but NOT those genera", m[m.phy_bact & ~m.gut])]:
    if len(s):
        print(f"  {lbl:44s} {len(s):2d} enzymes, {int(s.dead.sum())} inactive ({s.dead.mean():5.1%})")
