"""Full sequence vs mature region: does dropping the signal peptide help Horizyn?"""
from pathlib import Path
import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
pd.set_option("display.width", 200)

lab = lb.build(min_reps=1).rename(columns={"Enzyme":"accession","Amine":"amine","Hydroxyl":"core"})
res = {}
for v in ("full", "mature"):
    h = pd.read_csv(ROOT/f"outputs/horizyn_scores_{v}.csv")
    res[v] = h.merge(lab[["accession","amine","core","active"]], on=["accession","amine","core"])

print("=== can Horizyn predict which conjugations happen? ===")
print(f"  {'variant':10s} {'enzymes':>8s} {'cells':>7s} {'PR-AUC':>8s} {'ROC':>7s} {'chance PR':>10s}")
for v, m in res.items():
    y = m.active.astype(int)
    print(f"  {v:10s} {m.accession.nunique():8d} {len(m):7,d} "
          f"{average_precision_score(y, m.horizyn):8.4f} {roc_auc_score(y, m.horizyn):7.4f} "
          f"{y.mean():10.4f}")

print("\n=== head to head, on the 58 enzymes both cover ===")
common = set(res["full"].accession) & set(res["mature"].accession)
a = res["full"][res["full"].accession.isin(common)].sort_values(["accession","amine","core"])
b = res["mature"][res["mature"].accession.isin(common)].sort_values(["accession","amine","core"])
y = a.active.astype(int).to_numpy()
print(f"  {len(a):,} cells, {len(common)} enzymes, base rate {y.mean():.4f}")
for nm, s in (("full", a.horizyn.to_numpy()), ("mature", b.horizyn.to_numpy())):
    print(f"    {nm:7s} PR-AUC {average_precision_score(y,s):.4f}   ROC {roc_auc_score(y,s):.4f}")
print(f"  Spearman(full, mature) = {spearmanr(a.horizyn, b.horizyn).statistic:+.4f}")

print("\n=== product ranking vs what is actually made ===")
for v, m in res.items():
    p = m.groupby(["amine","core"]).agg(h=("horizyn","mean"), made=("active","mean")).reset_index()
    r, pv = spearmanr(p.h, p.made)
    print(f"  {v:10s} Spearman = {r:+.3f}  (p = {pv:.3f})")

print("\n=== per-core mean score vs how often that core is used ===")
for v, m in res.items():
    c = m.groupby("core").agg(horizyn=("horizyn","mean"), made=("active","mean"))
    print(f"  {v}:  " + "   ".join(f"{i} {r.horizyn:+.4f} (made {r.made:.1%})"
                                    for i, r in c.iterrows()))

print("\n=== within-substrate: fix the product, rank the enzymes ===")
for v, m in res.items():
    w = [roc_auc_score(s.active.astype(int), s.horizyn)
         for _, s in m.groupby(["amine","core"])
         if s.active.sum() >= 5 and (~s.active.astype(bool)).sum() >= 5]
    print(f"  {v:10s} mean AUC {np.mean(w):.4f} over {len(w)} products")
