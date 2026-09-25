"""Where does the model make a SPECIFIC claim rather than the generic one --
and when it does, is it right?"""
from pathlib import Path
import sys, itertools, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
pd.set_option("display.width", 210)
CUT = 0.5

g = pd.read_csv(ROOT/"outputs/candidate_predictions.csv")
lab = lb.build(min_reps=1)
prev = lab.groupby(["Amine","Hydroxyl"]).active.mean()
P = pd.DataFrame(__import__("json").loads((ROOT/"site/candidates.json").read_text())["proteins"])
org = dict(zip(P.accession, P.organism.fillna("").str.split().str[:2].str.join(" ")))
n = g.accession.nunique()
hit = g[g.raw_trimmed >= CUT]

print("=== 1. the discriminating calls: products predicted for SOME but not all ===")
t = hit.groupby(["Amine","Hydroxyl"]).accession.nunique().rename("proteins").reset_index()
t["share"] = t.proteins / n
t["made_by"] = [float(prev.get((a,h),0)) for a,h in zip(t.Amine,t.Hydroxyl)]
disc = t[(t.share > 0.02) & (t.share < 0.80)].sort_values("share", ascending=False)
print(disc.to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

print("\n=== 2. when it makes a rare call, is it right? ===")
m = g[g.measured.notna()].copy(); m["measured"] = m.measured.astype(int)
m["rarity"] = [float(prev.get((a,h),0)) for a,h in zip(m.Amine,m.Hydroxyl)]
for lo, hi, lbl in [(0,.15,"rare products (<15% make them)"),
                    (.15,.50,"uncommon (15-50%)"), (.50,1.01,"common (>50%)")]:
    s = m[(m.rarity>=lo)&(m.rarity<hi)]
    calls = s[s.raw_trimmed>=CUT]
    if not len(calls): continue
    print(f"  {lbl:34s} {len(calls):4d} calls, {calls.measured.mean():6.1%} correct "
          f"(base rate {s.measured.mean():.1%})")

print("\n=== 3. proteins that get an unusual prediction ===")
common = set(hit.Amine.value_counts().head(6).index)
odd = hit[~hit.Amine.isin(common)]
for acc, s in odd.groupby("accession"):
    meas = g[(g.accession==acc)&g.measured.notna()]
    tag = "never assayed" if not len(meas) else f"made {int(meas.measured.sum())} products"
    items = ", ".join(f"{r.Amine}+{r.Hydroxyl} ({r.raw_trimmed:.2f})" for r in s.itertuples())
    print(f"  {acc:12s} {org.get(acc,''):28s} {tag:16s} {items}")

print("\n=== 4. which proteins are least like the consensus? ===")
sets = {a: set(zip(s.Amine, s.Hydroxyl)) for a, s in hit.groupby("accession")}
cons = {p for p, c in pd.Series([x for s in sets.values() for x in s]).value_counts().items()
        if c >= 0.8*n}
rows = []
for a, s in sets.items():
    j = len(s & cons)/len(s | cons) if (s|cons) else 0
    rows.append(dict(accession=a, organism=org.get(a,""), n=len(s),
                     extra=len(s-cons), missing=len(cons-s), similarity=j))
d = pd.DataFrame(rows).sort_values("similarity")
print(f"  consensus set = {len(cons)} products predicted for >=80% of proteins")
print(d.head(10).to_string(index=False, float_format=lambda v: f"{v:,.3f}"))
