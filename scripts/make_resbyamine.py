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
cols, rows = pickle.load(open(f"{AF3}/analysis/all_contacts25.pkl", "rb"))
d = pd.DataFrame(rows, columns=cols)
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
     .groupby(["protein", "token", "resname", "resnum", "chain"])["sample"]
     .nunique().reset_index(name="f"))
g = g[g.f >= 3]
g["cls"] = g.token.map(tok2cls)

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
               n_contacts=int(len(g))),
          open("/home/adsiordia/BSH-Model/site/resbyamine.json", "w"),
          separators=(",", ":"))
print(f"wrote site/resbyamine.json")
print(f"  {len(g)} reproducible amine-side contacts, {len(ORDER)} classes, {len(keep)} residues")
print(f"  contacts per class: {dict(tot[ORDER])}")
