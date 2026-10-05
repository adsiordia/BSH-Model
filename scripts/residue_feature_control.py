#!/usr/bin/env python3
"""Is a pocket-residue feature set worth anything, or is it just lineage?

The decision of "which residues to use" should not be made by intuition or by
raw association with activity -- across these 115 enzymes every pocket position
shows a +2 to +6 conjugate raw difference, and almost all of it is ancestry
(see the site's "Residues & activity" tab).

This script makes the decision empirically, with two controls stacked:

  1. GroupKFold on the 50%-identity cluster, so every test fold is a lineage the
     model has never seen. Clade cannot be a feature under this split, which is
     the point -- it is what "will this work on a new sequence" means.
  2. A matched-random control. The pocket columns are compared against the same
     number of NON-contact alignment columns, drawn to match the pocket columns'
     entropy. Without the entropy match a more variable column wins on raw
     information content and the comparison says nothing about the pocket.

Result as of this writing: amine identity alone gives PR-AUC 0.739; the 7
variable amine-contacting positions give 0.798; entropy-matched random columns
give 0.768 +/- 0.023. So the pocket positions are worth +0.030 over matched
random (p = 0.10, 4 of 40 draws match or beat them) and roughly HALF the
apparent benefit of "using pocket residues" is generic sequence similarity.

Run with an env that has sklearn:
  /home/adsiordia/miniconda3/envs/pandamap/bin/python scripts/residue_feature_control.py
"""
import json, numpy as np, collections
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score

RA=json.load(open("site/resact.json")); AL=json.load(open("site/alignment.json"))
rows,amines,charge=RA["rows"],RA["amines"],RA["charge"]
ids=[r["id"] for r in rows]; aln=AL["aln127"]
contact127={p["col127"] for p in RA["positions"]}
L=len(aln[ids[0]])

def entropy(c):
    cnt=collections.Counter(aln[i][c] for i in ids if i in aln and aln[i][c]!="-")
    n=sum(cnt.values())
    if n<0.9*len(ids): return None
    p=np.array(list(cnt.values()))/n
    return float(-(p*np.log(p)).sum())

ent={c:entropy(c) for c in range(L)}
pocket=[p["col127"] for p in RA["positions"] if p["amine"] and not p["invariant"]]
pe=[ent[c] for c in pocket]
print(f"pocket columns: {len(pocket)}  entropies {[f'{x:.2f}' for x in pe]}")
cand=[c for c in range(L) if c not in contact127 and ent[c] is not None and ent[c]>0.05]
print(f"{len(cand)} non-contact columns with entropy > 0.05")

ai={a:i for i,a in enumerate(amines)}
def build(cols):
    AAs=sorted({aln[i][c] for i in ids if i in aln for c in cols}) if cols else []
    X,y,g=[],[],[]
    for r in rows:
        s=aln.get(r["id"],"")
        for a in amines:
            f=[0.0]*len(amines); f[ai[a]]=1.0
            f.append(1.0 if charge[a]!="neutral" else 0.0)
            for c in cols:
                oh=[0.0]*len(AAs)
                if s: oh[AAs.index(s[c])]=1.0
                f+=oh
            X.append(f);y.append(r["act"][a]);g.append(r["cluster"])
    return np.array(X),np.array(y),np.array(g)

def ap(cols,seed=0):
    X,y,g=build(cols); pred=np.zeros(len(y))
    for tr,te in GroupKFold(n_splits=5).split(X,y,groups=g):
        m=RandomForestClassifier(n_estimators=300,min_samples_leaf=3,
            random_state=seed,n_jobs=4,class_weight="balanced")
        m.fit(X[tr],y[tr]); pred[te]=m.predict_proba(X[te])[:,1]
    return average_precision_score(y,pred)

a0=ap([]); aP=ap(pocket)
print(f"\n  amine only                      {a0:.3f}")
print(f"  + the 7 pocket positions        {aP:.3f}   ({aP-a0:+.3f})")

# entropy-matched draws: for each pocket column pick a non-contact column of
# similar entropy, without replacement
rng=np.random.default_rng(1); draws=[]
for k in range(40):
    used=set(); pick=[]
    for e in pe:
        pool=sorted(cand,key=lambda c:abs(ent[c]-e))
        pool=[c for c in pool if c not in used][:25]
        c=int(rng.choice(pool)); used.add(c); pick.append(c)
    draws.append(ap(pick,seed=0))
D=np.array(draws)
print(f"  + 7 entropy-matched random cols  {D.mean():.3f} +/- {D.std():.3f}   ({D.mean()-a0:+.3f})")
print(f"\n  pocket advantage over matched random: {aP-D.mean():+.3f}")
print(f"  random draws >= pocket: {(D>=aP).sum()}/40   -> p = {(D>=aP).mean():.3f}")
print(f"  share of the gain that is pocket-specific: {(aP-D.mean())/(aP-a0):.0%}")
