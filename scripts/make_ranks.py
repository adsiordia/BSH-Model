#!/usr/bin/env python3
"""All five AlphaFold predictions per job, with each one's own metrics.

AF3 wrote 5 models per job and ranked them; everything already exists on disk
(3,830 per-sample model.cif, 3,375 PandaMap CSVs). Nothing is predicted or
re-run here -- this just collects, per job and per sample:

  rank          0 is the model AF3 ranked first
  rank_score    AF3's own ranking score
  ptm / iptm    complex confidence; iptm is the inter-chain one
  plddt_prot    mean pLDDT over the protein
  plddt_lig     mean pLDDT over the ligand -- the number that actually varies
  n_contacts    how many contacts PandaMap found in THAT model

so the viewer can offer any of the five and say what each one is worth.
"""
import csv, glob, json, os, collections
import numpy as np

AF3 = "/home/adsiordia/AF3/BSH_AF3"
SITE = "/home/adsiordia/BSH-Model/site"


def jobdir(job):
    for d in ("out_panel", "glytau/out", "round2/out"):
        p = f"{AF3}/{d}/{job}"
        if os.path.isdir(p):
            return p
    return None


jobs = sorted(os.path.basename(f)[:-4] for f in glob.glob(f"{SITE}/pockets/*.pdb"))
out = {}
for job in jobs:
    jd = jobdir(job)
    if not jd:
        continue
    rk = f"{jd}/{job}_ranking_scores.csv"
    if not os.path.exists(rk):
        continue
    rows = sorted(csv.DictReader(open(rk)),
                  key=lambda r: -float(r["ranking_score"]))
    ent = []
    for rank, r in enumerate(rows):
        sd = f"seed-{r['seed']}_sample-{r['sample']}"
        e = dict(rank=rank, sample=sd,
                 rank_score=round(float(r["ranking_score"]), 4))
        sc = f"{jd}/{sd}/{job}_{sd}_summary_confidences.json"
        if os.path.exists(sc):
            j = json.load(open(sc))
            for k in ("ptm", "iptm", "fraction_disordered"):
                if j.get(k) is not None:
                    e[k] = round(float(j[k]), 3)
            if j.get("has_clash") is not None:
                e["has_clash"] = bool(j["has_clash"])
        cif = f"{jd}/{sd}/{job}_{sd}_model.cif"
        if os.path.exists(cif):
            p, l = [], []
            for line in open(cif):
                if line.startswith("ATOM"):
                    p.append(float(line.split()[14]))
                elif line.startswith("HETATM"):
                    l.append(float(line.split()[14]))
            if p:
                e["plddt_prot"] = round(float(np.mean(p)), 1)
            if l:
                e["plddt_lig"] = round(float(np.mean(l)), 1)
        csvf = f"{AF3}/analysis/contacts/{job}__{sd}.csv"
        if os.path.exists(csvf):
            e["n_contacts"] = sum(1 for _ in open(csvf)) - 1
        ent.append(e)
    out[job] = ent

json.dump(out, open(f"{SITE}/ranks.json", "w"), separators=(",", ":"))
print(f"wrote site/ranks.json  ({os.path.getsize(f'{SITE}/ranks.json')/1024:.0f} KB)")
print(f"  {len(out)} jobs x {np.mean([len(v) for v in out.values()]):.0f} models each")
allr = [e for v in out.values() for e in v]
print(f"  {len(allr)} models in total\n")
print(f"  {'rank':>5}{'rank_score':>12}{'plddt_prot':>12}{'plddt_lig':>11}{'contacts':>10}")
for k in range(5):
    g = [e for e in allr if e["rank"] == k]
    f = lambda key: np.mean([e[key] for e in g if key in e])
    print(f"  {k:>5}{f('rank_score'):>12.4f}{f('plddt_prot'):>12.1f}"
          f"{f('plddt_lig'):>11.1f}{f('n_contacts'):>10.1f}")
lig = [[e["plddt_lig"] for e in v if "plddt_lig" in e] for v in out.values()]
best0 = sum(1 for v in lig if v and v[0] == max(v))
print(f"\n  Rank 0 also has the best ligand pLDDT in {best0} of {len(lig)} jobs "
      f"({100*best0/len(lig):.0f}%) — chance would be 20%")
