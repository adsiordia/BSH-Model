#!/usr/bin/env python3
"""Which alignment columns the pocket occupies, from the complete PandaMap output.

One threshold throughout: a contact counts when at least 3 of the 5 AlphaFold
samples show it, and a column counts as a pocket position when at least 3 enzymes
contact it that way. The shipped file used >=5 of 5 for the amine positions while
the alignment used >=3 and the By-enzyme table used >=1 -- three thresholds at
once, which is what made the tabs disagree.
"""
import json
import pandas as pd

AF3 = "/home/adsiordia/AF3/BSH_AF3"
SOLID, MIN_ENZ = 3, 3

d = pd.read_pickle(f"{AF3}/analysis/contacts_full.pkl")
d = d[d.protein != "A0A414Q275"]          # no catalytic cysteine
old = json.load(open(f"{AF3}/analysis/pocket_columns.json"))

out = {}
for part, key in (("moiety", "moiety"), ("core", "core")):
    x = d[d.part == part]
    f = (x.groupby(["protein", "token", "col", "pos"])["sample"]
         .nunique().reset_index(name="f"))
    f = f[f.f >= SOLID]
    n = f.groupby(["col", "pos"]).protein.nunique().reset_index(name="nenz")
    n = n[n.nenz >= MIN_ENZ]
    out[key] = sorted(int(c) for c in n.col.unique())
    out[key + "_ref"] = sorted(int(p) for p in n.pos.unique())

json.dump(out, open(f"{AF3}/analysis/pocket_columns.json", "w"))
print("rebuilt analysis/pocket_columns.json at >=3 of 5, >=3 enzymes\n")
for k in ("moiety_ref", "core_ref"):
    o = sorted(x for x in old[k] if x is not None)
    n = out[k]
    print(f"  {k:<12}{len(o):>4} -> {len(n)}")
    print(f"     added: {sorted(set(n)-set(o))}")
    print(f"     lost:  {sorted(set(o)-set(n)) or 'none'}")
