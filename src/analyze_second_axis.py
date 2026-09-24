"""Nestedness explains ~90%. Is there a SECOND axis in the other 10%?
Specifically: do the deviations cluster by amine chemistry?"""
from pathlib import Path
import sys, json, numpy as np, pandas as pd
from scipy.stats import fisher_exact
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
pd.set_option("display.width", 210)

lab = lb.build(min_reps=1)
M = lab.pivot_table(index="Enzyme", columns=["Amine","Hydroxyl"], values="active",
                    aggfunc="first").fillna(False).astype(int)
prev = M.mean(0).sort_values(ascending=False)
order = list(prev.index); Mo = M[order]
exp = np.zeros_like(Mo.values, dtype=bool)
for i, e in enumerate(Mo.index):
    exp[i, :int(Mo.loc[e].sum())] = True
resid = Mo.values.astype(bool) & ~exp          # the 141 "unexpected" calls

MONO = {"dopamine","tyramine","serotonin","3_methoxytyramine","tryptamine","2_aminophenol"}
amines = [c[0] for c in order]
is_mono = np.array([a in MONO for a in amines])

print("=== are the deviations concentrated in the monoamines? ===")
dev_mono  = int(resid[:, is_mono].sum());  dev_other = int(resid[:, ~is_mono].sum())
act = Mo.values.astype(bool)
act_mono  = int(act[:, is_mono].sum());    act_other = int(act[:, ~is_mono].sum())
print(f"  monoamine / catechol products : {int(is_mono.sum())} of {len(order)} columns")
print(f"    active cells {act_mono:4d}, of which unexpected {dev_mono:3d} ({dev_mono/max(act_mono,1):.1%})")
print(f"  everything else")
print(f"    active cells {act_other:4d}, of which unexpected {dev_other:3d} ({dev_other/max(act_other,1):.1%})")
_, p = fisher_exact([[dev_mono, act_mono-dev_mono],[dev_other, act_other-dev_other]])
print(f"  Fisher p = {p:.2e}")

print("\n=== does making one rare monoamine predict making another? ===")
rare_mono = [j for j,c in enumerate(order) if c[0] in MONO and prev[c] < 0.30]
print(f"  rare monoamine products (<30% prevalence): {len(rare_mono)}")
for j in rare_mono: print(f"    {order[j][0]:22s} {order[j][1]:5s} {prev[order[j]]:.1%}")
sub = Mo.values[:, rare_mono].astype(int)
cnt = sub.sum(1)
print(f"\n  enzymes making 0 of them: {int((cnt==0).sum())}")
print(f"  making 1                : {int((cnt==1).sum())}")
print(f"  making 2+               : {int((cnt>=2).sum())}")
exp_rate = sub.mean(0)
import itertools
co = []
for a, b in itertools.combinations(range(len(rare_mono)), 2):
    both = int(((sub[:,a]==1)&(sub[:,b]==1)).sum())
    expd = sub[:,a].sum()*sub[:,b].sum()/len(sub)
    co.append((order[rare_mono[a]], order[rare_mono[b]], both, expd))
C = pd.DataFrame([(f"{x[0][0]}+{x[0][1]}", f"{y[0]}+{y[1]}", b, e, b/e if e else np.nan)
                  for x,y,b,e in co], columns=["A","B","observed","expected","ratio"])
print("\n  strongest co-occurrences (observed vs chance):")
print(C.sort_values("ratio", ascending=False).head(8).to_string(
      index=False, float_format=lambda v: f"{v:,.2f}"))

print("\n=== who are the monoamine specialists? ===")
spec = pd.DataFrame({"breadth": Mo.sum(1).values, "rare_mono": cnt}, index=Mo.index)
spec["expected"] = spec.breadth / len(order) * len(rare_mono)
spec["excess"] = spec.rare_mono - spec.expected
print(spec.sort_values("excess", ascending=False).head(10).to_string(
      float_format=lambda v: f"{v:,.2f}"))
print(f"\n  correlation breadth vs rare-monoamine count: "
      f"r = {np.corrcoef(spec.breadth, spec.rare_mono)[0,1]:.3f}")
