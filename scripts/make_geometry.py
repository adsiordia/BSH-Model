#!/usr/bin/env python3
"""The one structural feature that has to be true for catalysis to happen.

Contacts turn out not to separate active enzymes from inactive ones for any
product once lineage is controlled (22 products, all collapse). But a contact is
a weak thing to ask for. BSH is an Ntn-hydrolase: the catalytic cysteine's
thiolate attacks the amide carbonyl of the conjugate. So the geometry that
matters is whether that carbonyl is actually presented to the nucleophile --

    d(Cys1 SG -> scissile carbonyl C)

which is a direct, mechanistic measure of catalytic competence rather than a
proxy. Nothing in the project has looked at it.

The scissile carbonyl is found the same way the amine/core split is: the amide
C-N bond that separates the ligand into a bile acid and an amine, located from
geometry alone. A0A414Q275 is skipped throughout -- its supplied sequence has no
cysteine at all.
"""
import gzip, glob, json, os, collections
import numpy as np

SITE = "/home/adsiordia/BSH-Model/site"
COV = {("C", "C"): 1.75, ("C", "N"): 1.65, ("C", "O"): 1.60, ("C", "S"): 1.95,
       ("N", "N"): 1.55, ("N", "O"): 1.55, ("O", "O"): 1.60, ("S", "S"): 2.20,
       ("N", "S"): 1.80, ("O", "S"): 1.60}
cut = lambda a, b: COV.get(tuple(sorted((a, b))), 1.75)


def parse(path):
    """chain A's Cys1 SG, and the ligand's atoms."""
    sg, lig = None, []
    for l in gzip.open(path, "rt"):
        if l.startswith("ATOM") and l[21] == "A" and int(l[22:26]) == 1 \
           and l[12:16].strip() == "SG":
            sg = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        elif l.startswith("HETATM"):
            lig.append((l[12:16].strip(), l[76:78].strip().upper(),
                        np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])))
    return sg, lig


def scissile(lig):
    """the amide carbonyl carbon that joins the bile acid to the amine"""
    n = len(lig)
    if not n:
        return None
    X = np.array([a[2] for a in lig])
    el = [a[1] for a in lig]
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    adj = collections.defaultdict(set)
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= cut(el[i], el[j]):
                adj[i].add(j); adj[j].add(i)
    best = None
    for i in range(n):
        if el[i] != "N":
            continue
        for j in list(adj[i]):
            if el[j] != "C":
                continue
            if not any(el[k] == "O" and D[j, k] < 1.32 for k in adj[j] if k != i):
                continue
            seen, st = {i}, [i]
            while st:
                u = st.pop()
                for v in adj[u]:
                    if {u, v} == {i, j}:
                        continue
                    if v not in seen:
                        seen.add(v); st.append(v)
            if j in seen or len(seen) >= n / 2:
                continue
            if best is None or len(seen) > len(best[1]):
                best = (j, seen)
    return None if best is None else X[best[0]]


if __name__ == "__main__":
    out = {}
    for f in sorted(glob.glob(f"{SITE}/raw/*.pdb.gz")):
        job = os.path.basename(f)[:-7]
        if job.startswith("A0A414Q275"):
            continue                      # no catalytic cysteine at all
        sg, lig = parse(f)
        if sg is None:
            continue
        c = scissile(lig)
        if c is None:
            continue
        out[job] = round(float(np.linalg.norm(sg - c)), 2)
    json.dump(out, open(f"{SITE}/geometry.json", "w"), separators=(",", ":"))
    v = np.array(list(out.values()))
    print(f"wrote site/geometry.json  ({len(out)} jobs)")
    print(f"  d(Cys1 SG -> scissile carbonyl C): mean {v.mean():.2f} A, "
          f"median {np.median(v):.2f}, range {v.min():.2f}-{v.max():.2f}")
    print(f"  within 4 A (plausibly attack-ready): {(v<=4).sum()} of {len(v)} "
          f"({100*(v<=4).mean():.0f}%)")
    print(f"  beyond 8 A (cannot react as posed):  {(v>8).sum()} ({100*(v>8).mean():.0f}%)")
