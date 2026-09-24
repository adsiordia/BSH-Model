"""Is amine acceptance NESTED (one promiscuity axis) or are there distinct
specificity classes? Nested means a narrow enzyme's products are a subset of a
broad one's. Deviations from nesting are where real specificity lives.
"""
from pathlib import Path
import sys, json, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
pd.set_option("display.width", 210)

lab = lb.build(min_reps=1)
M = lab.pivot_table(index="Enzyme", columns=["Amine","Hydroxyl"],
                    values="active", aggfunc="first").fillna(False).astype(int)
breadth = M.sum(1).sort_values(ascending=False)
prev = M.mean(0).sort_values(ascending=False)
print(f"{M.shape[0]} enzymes x {M.shape[1]} products, {int(M.values.sum()):,} active cells\n")

print("=== product difficulty ordering (how many enzymes make each) ===")
for (a,h), v in prev.head(10).items():   print(f"  {a:26s} {h:5s} {v:6.1%}")
print("  ...")
for (a,h), v in prev.tail(6).items():    print(f"  {a:26s} {h:5s} {v:6.1%}")

# --- nestedness: for each active cell, is it "expected" given the enzyme's breadth? ---
order = list(prev.index)
Mo = M[order]
exp_mask = np.zeros_like(Mo.values, dtype=bool)
for i, e in enumerate(Mo.index):
    k = int(Mo.loc[e].sum())
    exp_mask[i, :k] = True          # a perfectly nested enzyme makes the k commonest
obs = Mo.values.astype(bool)
unexp = obs & ~exp_mask             # makes something rarer than its breadth implies
missing = ~obs & exp_mask           # skips something commoner than its breadth implies
print(f"\n=== how nested is it? ===")
print(f"  active cells                                  : {int(obs.sum()):,}")
print(f"  'unexpected' -- rarer product than breadth implies : {int(unexp.sum()):,} "
      f"({unexp.sum()/obs.sum():.1%})")
print(f"  'holes'      -- skips a commoner product           : {int(missing.sum()):,}")
print(f"  a perfectly nested matrix would have 0 of each.")

# temperature-style score: 0 = perfectly nested, 1 = random
rng = np.random.default_rng(0)
null = []
for _ in range(200):
    P = np.zeros_like(obs)
    for i, k in enumerate(obs.sum(1)):
        P[i, rng.choice(obs.shape[1], k, replace=False)] = True
    null.append((P & ~exp_mask).sum())
z = (unexp.sum() - np.mean(null)) / np.std(null)
print(f"  random matrices with the same row sums give {np.mean(null):,.0f} +/- {np.std(null):,.0f}")
print(f"  observed {int(unexp.sum()):,}  ->  z = {z:.1f}  "
      f"({'far more nested than chance' if z < -3 else 'not clearly nested'})")

print("\n=== enzymes that break the pattern most (idiosyncratic specificity) ===")
score = pd.DataFrame({"breadth": obs.sum(1), "odd": unexp.sum(1), "holes": missing.sum(1)},
                     index=Mo.index)
score["odd_rate"] = score.odd / score.breadth.clip(lower=1)
top = score[score.breadth >= 4].sort_values("odd", ascending=False).head(10)
print(top.to_string())

print("\n=== what do the odd ones make that they 'shouldn't'? ===")
rare = []
for i, e in enumerate(Mo.index):
    for j in np.where(unexp[i])[0]:
        rare.append((e, order[j][0], order[j][1], prev.iloc[j]))
R = pd.DataFrame(rare, columns=["Enzyme","Amine","Hydroxyl","prevalence"])
t = R.groupby(["Amine","Hydroxyl"]).agg(n_enzymes=("Enzyme","size"),
                                        prevalence=("prevalence","first"))
print(t.sort_values("n_enzymes", ascending=False).head(14).to_string(
      float_format=lambda v: f"{v:.3f}"))

print("\n=== do the rare-product makers cluster? ===")
rare_cols = [c for c in order if prev[c] < 0.15]
sub = Mo[rare_cols]
makers = sub.sum(1)
print(f"  {len(rare_cols)} products are made by <15% of enzymes")
print(f"  enzymes making >=3 of them: {int((makers>=3).sum())} of {len(Mo)}")
print(f"  enzymes making none       : {int((makers==0).sum())}")
print(f"  correlation between breadth and rare-product count: "
      f"r = {np.corrcoef(breadth.reindex(makers.index), makers)[0,1]:.3f}")
