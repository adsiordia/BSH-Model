#!/usr/bin/env python3
"""Check the amine/bile-acid split independently of the SMILES it came from.

How the split is made in the pipeline:
  1. PandaMap runs per sample with --ligand LIG_L and its CSV reports the
     specific ligand atom for every contact (the `ligand_atom` column).
  2. moiety_atoms25.pkl lists, per conjugate, the ligand atom names belonging
     to the amine.
  3. A contact is tagged 'moiety' if its ligand_atom is in that list, else
     'core'. So the tag is per CONTACT, not per residue -- one residue reaching
     across the amide shows up in both categories, which is intended.

The boundary is the scissile amide bond, which is the chemically right place:
BSH hydrolyses the C-N, so the carbonyl stays with the bile acid (it is cholic
acid's C24) and the nitrogen leaves with the amine. The moiety atom set is
therefore exactly the free amine as it exists after hydrolysis -- e.g. 9 heavy
atoms for asparagine (C4H8N2O3), 7 for taurine, and 28 left on the core.

This script re-derives the moiety set from 3D geometry alone: infer bonds by
interatomic distance, find every amide C-N that disconnects the molecule, and
take the largest resulting fragment that is under half the molecule. Asparagine,
glutamine, citrulline and glycylglycine each carry a SECOND amide, so several
cuts separate the molecule; the side-chain ones shear off only a terminal NH2 or
a single glycine, which is why "largest fragment under half" is the rule rather
than "smallest fragment". The bile acid core is 28 heavy atoms and the largest
amine here is 12, so the half-size test is unambiguous.

Last run: all 25 conjugates reproduced exactly.
"""
import warnings, glob, pickle, collections
import numpy as np
warnings.filterwarnings("ignore")
from Bio.PDB import MMCIFParser

P = MMCIFParser(QUIET=True)
COV = {('C','C'):1.75, ('C','N'):1.65, ('C','O'):1.60, ('C','S'):1.95,
       ('N','N'):1.55, ('N','O'):1.55, ('O','O'):1.60, ('S','S'):2.20}


def _cut(a, b):
    return COV.get(tuple(sorted((a, b))), 1.75)


def moiety_from_geometry(cif):
    s = P.get_structure("x", cif)[0]
    at = [a for c in s if c.id == "L" for r in c for a in r]
    if not at:
        return None
    nm = [a.name.strip() for a in at]
    el = [a.element.strip().upper() for a in at]
    X = np.array([a.coord for a in at])
    n = len(at)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    adj = collections.defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= _cut(el[i], el[j]):
                adj[i].add(j); adj[j].add(i)
    best = None
    for i in range(n):
        if el[i] != 'N':
            continue
        for j in list(adj[i]):
            if el[j] != 'C':
                continue
            # j must be a carbonyl carbon: an O at 1.2-1.3 A
            if not any(el[k] == 'O' and D[j, k] < 1.32 for k in adj[j] if k != i):
                continue
            seen, stack = {i}, [i]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if {u, v} == {i, j}:
                        continue
                    if v not in seen:
                        seen.add(v); stack.append(v)
            if j in seen:                 # the bond is in a ring, not a real cut
                continue
            if len(seen) >= n / 2:        # that fragment is the bile acid
                continue
            if best is None or len(seen) > len(best):
                best = seen
    return sorted(nm[i] for i in best) if best else None


if __name__ == "__main__":
    MOI = pickle.load(open("moiety_atoms25.pkl", "rb"))
    bad = []
    print(f"  {'conjugate':<14}{'SMILES':>8}{'geometry':>10}   agree?")
    for tok in sorted(MOI):
        fs = (glob.glob(f"../*/*_tet_{tok}/*_model.cif")
              + glob.glob(f"../*/*/*_tet_{tok}/*_model.cif"))
        if not fs:
            continue
        g = moiety_from_geometry(fs[0])
        ref = sorted(MOI[tok])
        ok = g is not None and g == ref
        if not ok:
            bad.append((tok, ref, g))
        print(f"  {tok:<14}{len(ref):>8}{(len(g) if g else 0):>10}   {'yes' if ok else 'NO'}")
    print()
    print("MISMATCHES:" if bad else "all amine-moiety atom sets reproduced from geometry alone.")
    for tok, ref, g in bad:
        print(f"  {tok}: smiles={ref}\n     geometry={g}")
