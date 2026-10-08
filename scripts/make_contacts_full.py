#!/usr/bin/env python3
"""The complete PandaMap output, nothing dropped.

The dataset every analysis has run on (analysis/all_contacts25.pkl) kept 5 of
PandaMap's 13 interaction types and discarded 75,067 of 127,807 rows -- 59%.
The eight dropped types were alkyl_pi, carbon_pi, pi_pi_stacking, donor_pi,
amide_pi, pi_cation, cation_pi and covalent: seven of the eight are the pi /
aromatic interactions, and covalent matters in an enzyme with a covalent
cysteine mechanism.

This rebuild keeps every row and every column PandaMap wrote. Two columns are
ADDED, nothing is altered:

  part  amine-side or core-side, by whether the ligand atom belongs to the
        amine moiety -- the same geometric rule verified on all 25 conjugates
  pos   the residue's position in the C0ERS1 alignment frame, by the route
        verified 893/893 against the CA records of the shipped structures

No filtering, no thresholding, no protein excluded -- A0A414Q275 is kept here
and left for each analysis to exclude where its missing catalytic cysteine
matters.
"""
import csv, glob, json, os, pickle, re, collections
import pandas as pd

AF3 = "/home/adsiordia/AF3/BSH_AF3"
SITE = "/home/adsiordia/BSH-Model/site"

MOI = pickle.load(open(f"{AF3}/analysis/moiety_atoms25.pkl", "rb"))
AL = json.load(open(f"{SITE}/alignment.json"))
REF = {int(a): b for a, b in AL["ref127"].items()}
src = json.load(open(f"{AF3}/analysis/sequence_source.json"))

# mature-sequence index -> alignment column, per enzyme
idx = {}
for e, s in AL["aln127"].items():
    m, n = {}, 0
    for c, ch in enumerate(s):
        if ch != "-":
            n += 1
            m[n] = c
    idx[e] = m


def pos_of(prot, resnum):
    o = idx.get(prot)
    if not o:
        return None
    i = resnum + (1 if src.get(prot) == "raw, untouched" else 0)
    c = o.get(i)
    return REF.get(c) if c is not None else None


rows = []
files = sorted(glob.glob(f"{AF3}/analysis/contacts/*.csv"))
for f in files:
    b = os.path.basename(f)[:-4]
    job, samp = b.split("__")
    prot, token = job.split("_tet_")
    s = int(re.search(r"sample-(\d+)$", samp).group(1))
    seed = int(re.search(r"seed-(\d+)", samp).group(1))
    moi = MOI.get(token, ())
    for r in csv.DictReader(f_ := open(f)):
        rn = int(r["protein_resnum"])
        rows.append((job, prot, token, seed, s,
                     r["interaction_type"], r["protein_residue"], rn,
                     r["protein_chain"], r["protein_atom"],
                     r["ligand_atom"], r["ligand_element"],
                     float(r["distance_A"]),
                     r["solvent_accessible"], r["halogen_element"],
                     "moiety" if r["ligand_atom"] in moi else "core",
                     pos_of(prot, rn)))
    f_.close()

d = pd.DataFrame(rows, columns=[
    "job", "protein", "token", "seed", "sample", "itype", "resname", "resnum",
    "chain", "patom", "latom", "lelem", "dist", "solvent_accessible",
    "halogen_element", "part", "pos"])
d.to_pickle(f"{AF3}/analysis/contacts_full.pkl")

print(f"wrote analysis/contacts_full.pkl")
print(f"  {len(d)} rows from {len(files)} CSV files")
print(f"  {d.job.nunique()} jobs, {d.protein.nunique()} proteins, "
      f"{d.token.nunique()} conjugates, {d['sample'].nunique()} samples each")
print(f"  {d.itype.nunique()} interaction types, all kept\n")
old = pickle.load(open(f"{AF3}/analysis/all_contacts25.pkl", "rb"))
print(f"  previous dataset: {len(old[1])} rows, 5 types")
print(f"  this one:         {len(d)} rows, {d.itype.nunique()} types "
      f"(+{len(d)-len(old[1])} rows recovered)\n")
print("  rows by interaction type:")
for t, n in d.itype.value_counts().items():
    amine = (d[(d.itype == t)].part == "moiety").sum()
    print(f"    {t:<20}{n:>7}   amine-side {amine:>6} ({100*amine/n:>4.1f}%)")
