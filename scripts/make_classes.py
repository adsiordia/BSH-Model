#!/usr/bin/env python3
"""Class-level comparison: which pocket residues separate enzymes that are active
on a chemical CLASS of conjugate from those that are not.

Five classes, by the chemistry of the amine rather than its identity:

  aromatic      a ring on the amine        phenylalanine, tyramine, dopamine,
                                           3-methoxytyramine, tryptamine,
                                           serotonin, 2-aminophenol, histidine
  basic         a second amine/guanidine   arginine, lysine, ornithine, Dap
  polar         amide or hydroxyl side     asparagine, glutamine, citrulline,
                                           serine, threonine, cysteine
  small         small or aliphatic         alanine, glycylglycine, proline,
                                           methionine, GABA
  polyamine     no carboxylate at all      cadaverine, putrescine

Activity is scored on the Tri (cholic) core, because that is the core every AF3
structure uses. An enzyme counts as active on a class if it makes at least one
member of it.

The comparison is run twice: once naively, and once stratified by 50%-identity
cluster, keeping only clusters that hold at least two actives and two inactives.
The gap between the two is the point of the script.
"""
import json, collections, itertools
from scipy.stats import fisher_exact

SITE = "/home/adsiordia/BSH-Model/site"
CLASSES = {
    "aromatic": ["phenylalanine", "tyramine", "dopamine", "3_methoxytyramine",
                 "tryptamine", "serotonin", "2_aminophenol", "histidine"],
    "basic":    ["arginine", "lysine", "ornithine", "2,3_diaminopropionic acid"],
    "polar":    ["asparagine", "glutamine", "citrulline", "serine", "threonine",
                 "cystine"],
    "small":    ["alanine", "glyglycine", "proline", "methionine", "gaba"],
    "polyamine": ["cadaverine", "putrescine"],
}

cells = json.load(open(f"{SITE}/data.json"))["cells"]
RA = json.load(open(f"{SITE}/resact.json"))
cluster = {r["id"]: r["cluster"] for r in RA["rows"]}
res = {r["id"]: r["res"] for r in RA["rows"]}
POS = {str(p["col"]): p for p in RA["positions"]}

# active on the Tri core only -- the core the structures use
act = collections.defaultdict(set)
for c in cells:
    if c["Hydroxyl"] == "Tri" and c["active"]:
        act[c["Enzyme"]].add(c["Amine"])
enz = sorted(e for e in res if e in cluster)

out = {}
for cls, members in CLASSES.items():
    A = {e for e in enz if act[e] & set(members)}
    I = set(enz) - A
    rows = []
    for col, meta in POS.items():
        if meta["invariant"]:
            continue
        aas = {res[e][col] for e in enz if res[e][col] != "-"}
        for aa in sorted(aas):
            a = sum(1 for e in A if res[e][col] == aa)
            b = len(A) - a
            c = sum(1 for e in I if res[e][col] == aa)
            dd = len(I) - c
            if a + c < 4:
                continue
            odds, p = fisher_exact([[a, b], [c, dd]])
            # stratified: only clusters holding >=2 active and >=2 inactive
            num = den = 0
            nstrat = 0
            for cl in {cluster[e] for e in enz}:
                ea = [e for e in A if cluster[e] == cl]
                ei = [e for e in I if cluster[e] == cl]
                if len(ea) < 2 or len(ei) < 2:
                    continue
                n = len(ea) + len(ei)
                aa_a = sum(1 for e in ea if res[e][col] == aa)
                aa_i = sum(1 for e in ei if res[e][col] == aa)
                num += aa_a * (len(ei) - aa_i) / n
                den += (len(ea) - aa_a) * aa_i / n
                nstrat += 1
            rows.append(dict(pos=meta["ref"], col=col, aa=aa, part=("amine" if meta["amine"] else "core"),
                             n_act=a, n_inact=c, odds=round(odds, 2) if odds not in (0, float("inf")) else None,
                             p=round(p, 5), nstrat=nstrat,
                             mh=(round(num / den, 2) if den else None)))
    rows.sort(key=lambda r: r["p"])
    out[cls] = dict(n_active=len(A), n_inactive=len(I),
                    clusters_active=len({cluster[e] for e in A}),
                    clusters_inactive=len({cluster[e] for e in I}), rows=rows)

json.dump(dict(classes=CLASSES, result=out), open(f"{SITE}/classes.json", "w"),
          separators=(",", ":"))
print(f"{len(enz)} enzymes with sequence and Tri-core assay data\n")
print(f"  {'class':<11}{'active':>8}{'inactive':>10}{'clades(act)':>13}"
      f"{'top position':>16}{'naive p':>10}{'stratified':>12}")
for cls, r in out.items():
    t = r["rows"][0] if r["rows"] else None
    print(f"  {cls:<11}{r['n_active']:>8}{r['n_inactive']:>10}{r['clusters_active']:>13}"
          f"{(str(t['pos'])+t['aa']) if t else '-':>16}{t['p'] if t else 0:>10.5f}"
          f"{(str(t['mh'])+' ('+str(t['nstrat'])+' clades)') if t and t['mh'] is not None else 'none testable':>12}")
