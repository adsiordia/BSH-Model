"""The Bacillota enzymes: what they actually make, and what trimming does to them."""
from pathlib import Path
import sys, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
OUT = ROOT/"outputs"; CUT = 0.5; pd.set_option("display.width", 210)

lab = lb.build(min_reps=1)
P = pd.DataFrame(json.load(open(ROOT/"site/candidates.json"))["proteins"])
bac = set(P[P.phylum.fillna("")=="Bacillota"].accession)
g = pd.read_csv(OUT/"candidate_predictions.csv")
g = g[g.comparable & g.accession.isin(bac)].copy()
print(f"{g.accession.nunique()} Bacillota enzymes with both embeddings, {len(g):,} cells\n")

E = g.groupby("accession").agg(
    n_full=("raw_full", lambda s:int((s>=CUT).sum())),
    n_trim=("raw_trimmed", lambda s:int((s>=CUT).sum())),
    mean_full=("raw_full","mean"), mean_trim=("raw_trimmed","mean")).reset_index()
E["d_n"] = E.n_trim - E.n_full
E["d_mean"] = (E.mean_trim - E.mean_full).round(4)
truth = lab.groupby("Enzyme").active.sum()
E["measured"] = E.accession.map(truth)
E["organism"] = E.accession.map(dict(zip(P.accession, P.organism.fillna("").str.split().str[:2].str.join(" "))))

print("=== 1. did trimming raise any of them? ===")
print(E.sort_values("d_n", ascending=False)[
    ["accession","organism","measured","n_full","n_trim","d_n","mean_full","mean_trim","d_mean"]
    ].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
up, down, same = int((E.d_n>0).sum()), int((E.d_n<0).sum()), int((E.d_n==0).sum())
print(f"\n  gained products {up}   lost {down}   unchanged {same}   of {len(E)}")
print(f"  products predicted overall: {E.n_full.sum()} -> {E.n_trim.sum()} ({E.n_trim.sum()-E.n_full.sum():+d})")
st, pv = wilcoxon(E.mean_trim, E.mean_full)
print(f"  mean probability {E.mean_full.mean():.4f} -> {E.mean_trim.mean():.4f}"
      f"  Wilcoxon signed-rank p = {pv:.4f}")
big = E[E.d_n.abs() >= 3]
print(f"  enzymes moving by 3+ products: {len(big)}"
      + ("" if not len(big) else "\n" + big[['accession','organism','n_full','n_trim','d_n']].to_string(index=False)))

print("\n=== 2. what are they actually active on? (measured) ===")
mb = lab[lab.Enzyme.isin(bac)]
prod = mb[mb.active].groupby(["Amine","Hydroxyl"]).size().rename("enzymes").reset_index()
prod["of"] = mb.Enzyme.nunique()
prod["share"] = prod.enzymes/prod["of"]
allp = lab[lab.active].groupby(["Amine","Hydroxyl"]).size()/lab.Enzyme.nunique()
prod["all_enzymes"] = [float(allp.get((a,h),0)) for a,h in zip(prod.Amine,prod.Hydroxyl)]
prod["diff"] = prod.share - prod.all_enzymes
print(prod.sort_values("enzymes", ascending=False).head(18).to_string(
    index=False, float_format=lambda v: f"{v:,.3f}"))
print(f"\n  Bacillota make a median of {mb.groupby('Enzyme').active.sum().median():.0f} products"
      f"  (all enzymes: {lab.groupby('Enzyme').active.sum().median():.0f})")

print("\n=== 3. which products did trimming ADD or REMOVE for them? ===")
add = g[(g.raw_full<CUT)&(g.raw_trimmed>=CUT)]
rem = g[(g.raw_full>=CUT)&(g.raw_trimmed<CUT)]
for nm, d in (("ADDED by trimming", add), ("REMOVED by trimming", rem)):
    print(f"\n  {nm}: {len(d)} calls")
    if len(d):
        t = d.groupby(["Amine","Hydroxyl"]).agg(n=("Amine","size"),
              measured_active=("measured", lambda s: int(s.fillna(0).sum()))).reset_index()
        t["made_by_all"] = [float(allp.get((a,h),0)) for a,h in zip(t.Amine,t.Hydroxyl)]
        print(t.sort_values("n", ascending=False).to_string(
            index=False, float_format=lambda v: f"{v:,.3f}"))
