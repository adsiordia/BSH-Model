"""The reverse question: does trimming turn ACTIVE into INACTIVE, and by how much?"""
from pathlib import Path
import sys, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
OUT = ROOT/"outputs"; CUT = 0.5; pd.set_option("display.width", 200)

g = pd.read_csv(OUT/"candidate_predictions.csv")
g = g[g.comparable]
lab = lb.build(min_reps=1); truth = lab.groupby("Enzyme").active.sum()
g["assay_active"] = g.accession.map(truth)
m = g[g.measured.notna()].copy(); m["measured"] = m.measured.astype(int)

print("=== 1. PROTEIN level: do active enzymes go silent when trimmed? ===")
P = g.groupby("accession").agg(n_full=("raw_full", lambda s:int((s>=CUT).sum())),
                               n_trim=("raw_trimmed", lambda s:int((s>=CUT).sum()))).reset_index()
P["truth"] = P.accession.map(truth); P = P[P.truth.notna()]
A = P[P.truth > 0]
print(f"  {len(A)} enzymes active in the assay")
print(f"    predicted silent (0 products) on FULL    : {int((A.n_full==0).sum())}")
print(f"    predicted silent (0 products) on TRIMMED : {int((A.n_trim==0).sum())}")
print(f"    lost at least one product                : {int((A.n_trim<A.n_full).sum())} of {len(A)}")
print(f"    gained at least one                      : {int((A.n_trim>A.n_full).sum())} of {len(A)}")
print(f"    products predicted: {A.n_full.sum()} -> {A.n_trim.sum()} "
      f"({A.n_trim.sum()-A.n_full.sum():+d}, {A.n_trim.sum()/A.n_full.sum()-1:+.1%})")

print("\n=== 2. CELL level: real, measured products the model stops calling ===")
ma = m[m.measured == 1]
kept = int(((ma.raw_full>=CUT) & (ma.raw_trimmed>=CUT)).sum())
lost = int(((ma.raw_full>=CUT) & (ma.raw_trimmed< CUT)).sum())
never = int(((ma.raw_full< CUT) & (ma.raw_trimmed< CUT)).sum())
regain = int(((ma.raw_full< CUT) & (ma.raw_trimmed>=CUT)).sum())
print(f"  {len(ma)} cells were MEASURED active")
print(f"    called by both versions          : {kept}")
print(f"    called on full, LOST when trimmed: {lost}   <-- the loss")
print(f"    missed by full, caught by trimmed: {regain}")
print(f"    missed by both                   : {never}")
print(f"\n  recall on real products: {(kept+lost)/len(ma):.1%} (full) -> "
      f"{(kept+regain)/len(ma):.1%} (trimmed)   a drop of "
      f"{((kept+lost)-(kept+regain))/len(ma):.1%}")
print(f"  mean probability on measured-active cells: {ma.raw_full.mean():.3f} -> "
      f"{ma.raw_trimmed.mean():.3f}  ({ma.raw_trimmed.mean()-ma.raw_full.mean():+.3f})")

print("\n=== 3. WHICH products get lost? ===")
L = ma[(ma.raw_full>=CUT) & (ma.raw_trimmed<CUT)]
prev = lab.groupby(["Amine","Hydroxyl"]).active.mean()
t = L.groupby(["Amine","Hydroxyl"]).agg(lost=("Amine","size"),
        mean_full=("raw_full","mean"), mean_trim=("raw_trimmed","mean")).reset_index()
t["made_by"] = [prev.get((a,h), np.nan) for a,h in zip(t.Amine,t.Hydroxyl)]
t = t.sort_values("lost", ascending=False)
print(t.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
print(f"\n  median prevalence of a LOST product : {t.made_by.median():.1%}")
K = ma[(ma.raw_full>=CUT)&(ma.raw_trimmed>=CUT)].groupby(["Amine","Hydroxyl"]).size()
kp = prev.loc[list(K.index)]
print(f"  median prevalence of a KEPT product : {kp.median():.1%}")
print(f"  median prevalence, all 75 products  : {prev.median():.1%}")

print("\n=== 4. the two directions side by side ===")
dead_g = int(((m.measured==0) & (m.raw_full<CUT) & (m.raw_trimmed>=CUT)).sum())
print(f"  measured-INACTIVE cells that trimming turns ON  : {dead_g}")
print(f"  measured-ACTIVE  cells that trimming turns OFF  : {lost}")
print(f"  net effect on agreement with the assay: "
      f"{-(dead_g+lost)+regain:+d} cells")
