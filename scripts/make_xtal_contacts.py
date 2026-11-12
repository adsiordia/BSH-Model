#!/usr/bin/env python3
"""Rebuild the crystal contact files, keeping PandaMap's atom pair.

The distance shown anywhere on the site is PandaMap's, between the two atoms
that actually make the interaction. To draw that as a line in 3D the atom names
are needed, not just the distance, so they are carried through here.

8BLT: one ligand (taurocholate), split into amine/core with the geometric rule.
2BJF: two ligands (taurine, deoxycholate) already separate, so the taurine rows
      are amine-subsite contacts by construction.
7SVG: same shape as 2BJF -- taurine plus chenodeoxycholate, already separate.
      L. gasseri BSHA at 1.35 A with all four chains deposited, so unlike 2BJF
      no symmetry rebuild is needed. Its catalytic cysteine is oxidised to OCS,
      as 8BLT's is. The deposit holds THREE CDCA copies and one of them (JN3 503
      in chain A) sits 13.9 A from the nearest catalytic OCS CA, i.e. a surface
      site rather than the core subsite, so it is excluded here -- keeping it
      adds four positions seen in only one copy.
"""
import json, csv, glob, os, sys, warnings
warnings.filterwarnings("ignore")
from Bio.PDB import MMCIFParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _paths import REFSTRUCT, SITE

os.chdir(REFSTRUCT)          # every path below is relative to the tracked inputs
AL = json.load(open(SITE / "alignment.json"))
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

# ---- 7SVG: products already separate, like 2BJF ----
ref7 = res2ref("7SVG.cif", "7SVG_added.aln", "7SVG_LgBSHA")
# collect() keys ligands by chain only, but 7SVG has two CDCA copies in chain A
# and one of them is the surface site, so the copy is read from the filename
r7 = []
for f in sorted(glob.glob("pm7/7SVG_*.csv")):
    lig, cc = os.path.basename(f)[:-4].split("_")[1:3]
    if lig == "JN3" and cc == "A503":          # surface site, 13.9 A from OCS
        continue
    for r in csv.DictReader(open(f)):
        rn = int(r["protein_resnum"])
        r7.append(dict(lig=lig, ligchain=cc[0], lignum=int(cc[1:]),
                       chain=r["protein_chain"], resnum=rn,
                       resname=r["protein_residue"], itype=r["interaction_type"],
                       pa=r["protein_atom"], la=r["ligand_atom"],
                       dist=float(r["distance_A"]), ref=ref7.get(rn),
                       itf=r["protein_chain"] != cc[0]))
json.dump(r7, open("7SVG_contacts.json", "w"))

# ---- 2BJF: products already separate ----
ref2 = res2ref("2BJF.cif", "2BJF_added.aln", "2BJF_CpCBAH")
r2 = collect("pm2/2BJF_*.csv",
             lambda b: (b.split("_")[1], b.split("_")[2]), ref2)
json.dump(r2, open("2BJF_contacts.json", "w"))

print(f"8BLT: {len(r8)} contacts, "
      f"{sum(1 for r in r8 if r['part']=='moiety')} amine-side")
print(f"2BJF: {len(r2)} contacts, "
      f"{sum(1 for r in r2 if r['lig']=='TAU')} taurine")
print(f"7SVG: {len(r7)} contacts, "
      f"{sum(1 for r in r7 if r['lig']=='TAU')} taurine")
print("atom pairs kept:", all("pa" in r and "la" in r for r in r8 + r2 + r7))
