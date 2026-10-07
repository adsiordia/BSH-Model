#!/usr/bin/env python3
"""Which amines does each residue-at-a-position actually contact?

Rows are (alignment position, residue identity) -- "an arginine at 206", "a
tyrosine at 19" -- and columns are the 25 conjugates. A cell is the fraction of
the enzymes carrying that residue there whose structures put it in reproducible
contact with that conjugate's amine (>=3 of 5 AlphaFold samples, amine side
only).

This is a within-enzyme measurement pooled afterwards, so the contact itself is
not lineage-confounded. Comparing two residues at the same position still is,
and the cluster count per row is carried so the tab can say so.
"""
import pickle, json, collections
import pandas as pd

AF3 = "/home/adsiordia/AF3/BSH_AF3"
SITE = "/home/adsiordia/BSH-Model/site"

cols, rows = pickle.load(open(f"{AF3}/analysis/all_contacts25.pkl", "rb"))
d = pd.DataFrame(rows, columns=cols)
d = d[d.protein != "A0A414Q275"]          # no catalytic Cys, excluded from claims
charge = json.load(open(f"{AF3}/analysis/amine_charge.json"))
c2r = pickle.load(open(f"{AF3}/analysis/msa_struct.pkl", "rb"))["col2res"]["C0ERS1"]
PC = json.load(open(f"{AF3}/analysis/pocket_columns.json"))
RA = json.load(open(f"{SITE}/resact.json"))
cluster = {r["id"]: r["cluster"] for r in RA["rows"]}

AROM = {"Phe", "Tyramine", "Dopamine", "3MeOTyramine", "Tryptamine",
        "2Aminophenol", "His"}
# chemical classes of the amine, which is the altitude the claims can support:
# 25 individual conjugates give per-cell counts too small to read a trend off.
CLS = {"aromatic":  ["Phe", "Tyramine", "Dopamine", "3MeOTyramine",
                     "Tryptamine", "2Aminophenol", "His"],
       "basic":     ["Arg", "Orn", "Dap"],
       "polar":     ["Asn", "Gln", "Cit", "Ser", "Thr", "Cys"],
       "small":     ["Ala", "GlyGly", "Pro", "Met", "GABA", "Gly"],
       "polyamine": ["Cadaverine", "Putrescine"],
       "sulfonate": ["Tau"]}
AA3 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
       "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
       "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
       "TYR": "Y", "VAL": "V"}

AM = sorted(d.token.unique())
PR = sorted(d.protein.unique())

g = (d[d.part == "moiety"].groupby(["protein", "token", "col"])["sample"]
     .nunique().reset_index(name="f"))
hit = {(r.protein, r.token, r.col) for r in g[g.f >= 3].itertuples()}

ident = {}
for (p, col), x in d.groupby(["protein", "col"]):
    ident[(p, col)] = AA3.get(x.resname.mode().iat[0], "X")

out = []
for col in PC["moiety"]:
    pos = c2r.get(col)
    byaa = collections.defaultdict(list)
    for p in PR:
        aa = ident.get((p, col))
        if aa:
            byaa[aa].append(p)
    for aa, ps in byaa.items():
        if len(ps) < 3:
            continue
        rates = {a: round(sum((p, a, col) in hit for p in ps) / len(ps), 3)
                 for a in AM}
        mean = lambda sel: sum(rates[a] for a in sel) / max(1, len(sel))
        ch = [a for a in AM if charge.get(a, "neutral") != "neutral"]
        nu = [a for a in AM if charge.get(a, "neutral") == "neutral"]
        ar = [a for a in AM if a in AROM]
        na = [a for a in AM if a not in AROM]
        cls = {k: round(sum(rates[a] for a in v if a in rates)
                        / max(1, len([a for a in v if a in rates])), 3)
               for k, v in CLS.items()}
        # polarity: aromatic preference minus the mean of the carboxylate classes
        pol = round(cls["aromatic"]
                    - sum(cls[k] for k in ("basic", "polar", "small")) / 3, 3)
        out.append(dict(cls=cls, pol=pol, pos=pos, col=col, aa=aa, n=len(ps),
                        clusters=len({cluster.get(p) for p in ps if p in cluster}),
                        enzymes=sorted(ps), rates=rates,
                        chg=round(mean(ch), 3), neu=round(mean(nu), 3),
                        arom=round(mean(ar), 3), nonarom=round(mean(na), 3),
                        sel=round(mean(ch) - mean(nu), 3),
                        asel=round(mean(ar) - mean(na), 3)))

out.sort(key=lambda r: -max(abs(r["sel"]), abs(r["asel"])))
# the equivalent residue in the two published structures, so a position quoted
# in the C0ERS1 frame can be looked up in the PDB by someone outside the project
XF = json.load(open(f"{AF3}/refstruct/xtal_frames.json"))
for r in out:
    b = XF["2BJF"].get(str(r["pos"])); l = XF["8BLT"].get(str(r["pos"]))
    r["x2BJF"] = f"{b[1]}{b[0]}" if b else None
    r["x8BLT"] = f"{l[1]}{l[0]}" if l else None

# What KIND of interaction is being counted at each position+residue. PandaMap's
# labels are not five physical categories: attractive_charge and ionic are the
# same rows, and repulsion is UNFAVOURABLE yet still reported as an interaction.
# Position 268 is 39% repulsion on charged conjugates -- a carboxylate residue
# near a carboxylate amine -- so its "charge selectivity" is partly proximity,
# not attraction. The arginines at 222 and 206 are 0% and 5%.
mo = d[d.part == "moiety"]
ity = {}
for (col, rn), x in mo.groupby(["col", "resname"]):
    aa = AA3.get(rn, "X")
    n = len(x)
    if n < 40:
        continue
    ity[f"{col}:{aa}"] = {t: round(100 * (x.itype == t).sum() / n)
                          for t in ("hydrogen_bonds", "attractive_charge",
                                    "hydrophobic", "repulsion")}
    ity[f"{col}:{aa}"]["n"] = int(n)
for r in out:
    r["itypes"] = ity.get(f'{r["col"]}:{r["aa"]}')

out.sort(key=lambda r: -r["pol"])
json.dump(dict(rows=out, amines=AM, charge=charge, classes=CLS,
               aromatic=sorted(AROM & set(AM)), n_enzymes=len(PR)),
          open(f"{SITE}/resamine.json", "w"), separators=(",", ":"))
print(f"wrote site/resamine.json: {len(out)} (position, residue) groups, "
      f"{len(AM)} conjugates, {len(PR)} enzymes")
for r in out[:6]:
    print(f"  {r['pos']}{r['aa']} n={r['n']} clusters={r['clusters']}  "
          f"charge {r['sel']:+.2f}  aromatic {r['asel']:+.2f}")
