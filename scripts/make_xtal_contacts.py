#!/usr/bin/env python3
"""Rebuild the crystal contact files, keeping PandaMap's atom pair.

The distance shown anywhere on the site is PandaMap's, between the two atoms
that actually make the interaction. To draw that as a line in 3D the atom names
are needed, not just the distance, so they are carried through here.

8BLT: one ligand (taurocholate), split into amine/core with the geometric rule.
2BJF: two ligands (taurine, deoxycholate) already separate, so the taurine rows
      are amine-subsite contacts by construction.
"""
import json, csv, glob, os, warnings
warnings.filterwarnings("ignore")
from Bio.PDB import MMCIFParser

AL = json.load(open("/home/adsiordia/BSH-Model/site/alignment.json"))
REF127 = {int(a): b for a, b in AL["ref127"].items()}
P = MMCIFParser(QUIET=True)


def res2ref(cif, alnfile, name, chain_index=0):
    s = P.get_structure("x", cif)[0]
    ch = list(s)[chain_index]
    order = sorted(r.id[1] for r in ch
                   if r.id[0] == " " or r.resname == "OCS")
    seqs, n = {}, None
    for l in open(alnfile):
        l = l.strip()
        if l.startswith(">"):
            n = l[1:]; seqs[n] = ""
        elif n:
            seqs[n] += l
    aln = seqs[name]
    m, k = {}, 0
    for col, c in enumerate(aln):
        if c != "-":
            m[order[k]] = REF127.get(col); k += 1
    return m


def collect(pattern, parse, ref):
    rows = []
    for f in sorted(glob.glob(pattern)):
        lig, lch = parse(os.path.basename(f)[:-4])
        for r in csv.DictReader(open(f)):
            rn = int(r["protein_resnum"])
            rows.append(dict(lig=lig, ligchain=lch, chain=r["protein_chain"],
                             resnum=rn, resname=r["protein_residue"],
                             itype=r["interaction_type"],
                             pa=r["protein_atom"], la=r["ligand_atom"],
                             dist=float(r["distance_A"]), ref=ref.get(rn),
                             itf=r["protein_chain"] != lch))
    return rows


# ---- 8BLT: one ligand, geometric amine/core split ----
ref8 = res2ref("8BLT.cif", "8BLT_added.aln", "8BLT_LsBSH")
moi = set(json.load(open("8BLT_tch_moiety.json")))
r8 = collect("pm/8BLT_*.csv", lambda b: ("TCH", b.split("_")[1][0]), ref8)
for r in r8:
    r["part"] = "moiety" if r["la"] in moi else "core"
json.dump(r8, open("8BLT_contacts.json", "w"))

# ---- 2BJF: products already separate ----
ref2 = res2ref("2BJF.cif", "2BJF_added.aln", "2BJF_CpCBAH")
r2 = collect("pm2/2BJF_*.csv",
             lambda b: (b.split("_")[1], b.split("_")[2]), ref2)
json.dump(r2, open("2BJF_contacts.json", "w"))

print(f"8BLT: {len(r8)} contacts, "
      f"{sum(1 for r in r8 if r['part']=='moiety')} amine-side")
print(f"2BJF: {len(r2)} contacts, "
      f"{sum(1 for r in r2 if r['lig']=='TAU')} taurine")
print("atom pairs kept:", all("pa" in r and "la" in r for r in r8 + r2))
