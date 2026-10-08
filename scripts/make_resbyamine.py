#!/usr/bin/env python3
"""Which KINDS of residue contact which KINDS of amine, aggregated over positions.

The Residue-to-amine grid asks the question per position. This asks it per
residue chemistry: pool every reproducible amine-side contact and ask, for each
class of amine, which amino acids line it more or less often than they line the
pocket overall.

Enrichment = that residue's share of contacts within the class, over its share
across all classes. 1.0 means exactly as expected, so this is a diverging
measure and is drawn on a diverging scale with a neutral midpoint.

Counts are carried alongside, because enrichment on a small class is unstable:
taurine has 75 contacts and the polyamines 69, against 645 for the polar class.
"""
import pickle, json, collections
import pandas as pd, numpy as np

AF3 = "/home/adsiordia/AF3/BSH_AF3"
# the COMPLETE PandaMap output: all 13 interaction types, nothing dropped
d = pd.read_pickle(f"{AF3}/analysis/contacts_full.pkl")
d = d[d.protein != "A0A414Q275"]

CLS = {"aromatic": ["Phe", "Tyramine", "Dopamine", "3MeOTyramine", "Tryptamine",
                    "2Aminophenol", "His"],
       "basic": ["Arg", "Orn", "Dap"],
       "polar": ["Asn", "Gln", "Cit", "Ser", "Thr", "Cys"],
       "small": ["Ala", "GlyGly", "Pro", "Met", "GABA", "Gly"],
       "sulfonate": ["Tau"],
       "polyamine": ["Cadaverine", "Putrescine"]}
tok2cls = {t: k for k, v in CLS.items() for t in v}

g = (d[d.part == "moiety"]
     .groupby(["protein", "token", "resname", "resnum", "chain", "col"])["sample"]
     .nunique().reset_index(name="f"))
g = g[g.f >= 3]
g["cls"] = g.token.map(tok2cls)

# Which positions does each residue type pool over? The chemistry view loses
# the position, and without this it looks like an independent line of evidence
# when it is the same eleven positions relabelled by what sits in them: 99% of
# the arginine contacts are positions 222, 206 and 15; 100% of the cysteine ones
# are position 1, the catalytic residue; 88% of the proline ones are position 80.
c2r = pickle.load(open(f"{AF3}/analysis/msa_struct.pkl", "rb"))["col2res"]["C0ERS1"]
PCJ = json.load(open(f"{AF3}/analysis/pocket_columns.json"))
NAMED = set(PCJ["moiety_ref"])
g["pos"] = g.col.map(lambda c: c2r.get(c))
bypos = {}
for rn, x in g.groupby("resname"):
    cnt = collections.Counter(int(p) for p in x.pos.dropna())
    tot = sum(cnt.values()) or 1
    bypos[rn] = dict(
        total=int(len(x)),
        named=round(100 * sum(n for p, n in cnt.items() if p in NAMED) / tot),
        top=[[int(p), int(n), round(100 * n / tot)] for p, n in cnt.most_common(5)])

tab = g.groupby(["cls", "resname"]).size().unstack(fill_value=0)
tot = tab.sum(axis=1)
frac = tab.div(tot, axis=0)
bg = g.groupby("resname").size() / len(g)
enr = (frac / bg).replace([np.inf, -np.inf], np.nan).fillna(0)

ORDER = [c for c in ["aromatic", "basic", "polar", "small", "sulfonate",
                     "polyamine"] if c in tab.index]
# keep residues with enough contacts overall to be worth plotting
keep = [r for r in tab.columns if tab[r].sum() >= 12]
keep.sort(key=lambda r: -tab[r].sum())

json.dump(dict(classes=ORDER, residues=keep,
               members={k: CLS[k] for k in ORDER},
               counts={c: {r: int(tab.loc[c, r]) for r in keep} for c in ORDER},
               enr={c: {r: round(float(enr.loc[c, r]), 3) for r in keep} for c in ORDER},
               totals={c: int(tot[c]) for c in ORDER},
               bg={r: round(float(bg[r]), 4) for r in keep},
               bypos={r: bypos[r] for r in keep},
               n_contacts=int(len(g))),
          open("/home/adsiordia/BSH-Model/site/resbyamine.json", "w"),
          separators=(",", ":"))
print(f"wrote site/resbyamine.json")
print(f"  {len(g)} reproducible amine-side contacts, {len(ORDER)} classes, {len(keep)} residues")
print(f"  contacts per class: {dict(tot[ORDER])}")
