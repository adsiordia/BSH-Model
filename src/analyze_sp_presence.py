"""The cleanest test of the peptidase hypothesis: do enzymes WITH a signal
peptide fail more often than enzymes WITHOUT one? Uses all 115 assayed enzymes."""
from pathlib import Path
import sys, h5py, numpy as np, pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
pd.set_option("display.width", 200)

lab = lb.build(min_reps=1)
n = lab.groupby("Enzyme").active.sum().rename("n_active").reset_index()
n.columns = ["accession", "n_active"]
with h5py.File(ROOT/"data/embeddings/proteins_without_signal_peptides_trimmed.h5") as f:
    sp = {k.split("_")[-1] for k in f.keys()}
n["has_sp"] = n.accession.isin(sp)
n["dead"] = n.n_active == 0
print(f"{len(n)} assayed enzymes: {int(n.has_sp.sum())} carry a signal peptide, "
      f"{int((~n.has_sp).sum())} do not\n")

print("=== if uncleaved signal peptides blocked activity, the WITH group should fail more ===")
for lbl, s in [("WITH a signal peptide", n[n.has_sp]), ("WITHOUT one", n[~n.has_sp])]:
    print(f"  {lbl:24s} {len(s):3d} enzymes, {int(s.dead.sum()):2d} made nothing "
          f"({s.dead.mean():5.1%}), median {s.n_active.median():.0f} products")
tab = [[int(n[n.has_sp].dead.sum()), int((~n[n.has_sp].dead).sum())],
       [int(n[~n.has_sp].dead.sum()), int((~n[~n.has_sp].dead).sum())]]
odds, p = fisher_exact(tab)
print(f"\n  Fisher exact p = {p:.4f}   odds ratio = {odds:.2f}")
print(f"  -> {'SUPPORTED' if p < 0.05 and tab[0][0]/sum(tab[0]) > tab[1][0]/sum(tab[1]) else 'NOT SUPPORTED'}")

print("\n=== and among the enzymes that DO work, is breadth affected? ===")
a, b = n[n.has_sp & ~n.dead].n_active, n[~n.has_sp & ~n.dead].n_active
_, p2 = mannwhitneyu(a, b)
print(f"  with a signal peptide : median {a.median():.0f} products (n={len(a)})")
print(f"  without               : median {b.median():.0f} products (n={len(b)})")
print(f"  Mann-Whitney p = {p2:.3f}")

print("\n=== the 14 enzymes that made nothing, split by signal peptide ===")
d = n[n.dead]
print(f"  {int(d.has_sp.sum())} have a signal peptide, {int((~d.has_sp).sum())} do not")
print("  with:   ", ", ".join(sorted(d[d.has_sp].accession)))
print("  without:", ", ".join(sorted(d[~d.has_sp].accession)))
