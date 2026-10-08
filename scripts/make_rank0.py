#!/usr/bin/env python3
"""The Rank 0 model per job, with its own metrics -- the single source of truth.

AlphaFold 3 writes 5 models per run and orders them by its internal ranking
score. Rank 0 is the top one and is what the top-level {job}_model.cif is a copy
of. Everything downstream now uses Rank 0 alone: one model, its PandaMap run,
its own confidence numbers. No sample unions, no reproducibility thresholds.

What changes by doing this, stated so it is not discovered later: the 5-sample
consensus was filtering contacts, and Rank 0 keeps every contact PandaMap found
in that one model. Contact counts therefore rise, and the "n of 5" reproducibility
measure no longer exists -- a contact either is or is not in the model.

Metrics carried per job: AF3's ranking score, pTM, ipTM, fraction disordered,
clash flag, and mean pLDDT over protein and ligand separately.
"""
import csv, glob, json, os, collections
import numpy as np

AF3 = "/home/adsiordia/AF3/BSH_AF3"
SITE = "/home/adsiordia/BSH-Model/site"


def job_dir(job):
    for d in ("out_panel", "glytau/out", "round2/out"):
        p = f"{AF3}/{d}/{job}"
        if os.path.isdir(p):
            return p
    return None


def rank0(job):
    """the sample AF3 ranked first, and its scores"""
    d = job_dir(job)
    if not d:
        return None
    f = f"{d}/{job}_ranking_scores.csv"
    if not os.path.exists(f):
        return None
    rows = list(csv.DictReader(open(f)))
    if not rows:
        return None
    srt = sorted(rows, key=lambda r: -float(r["ranking_score"]))
    top = srt[0]
    sd = f"seed-{top['seed']}_sample-{top['sample']}"
    out = dict(sample=sd, rank_score=round(float(top["ranking_score"]), 4),
               n_models=len(rows),
               margin=round(float(top["ranking_score"])
                            - float(srt[1]["ranking_score"]), 4) if len(srt) > 1 else None)
    sc = f"{d}/{sd}/{job}_{sd}_summary_confidences.json"
    if os.path.exists(sc):
        j = json.load(open(sc))
        for k in ("ptm", "iptm", "fraction_disordered", "has_clash"):
            if j.get(k) is not None:
                out[k] = round(j[k], 4) if isinstance(j[k], float) else j[k]
    cif = f"{d}/{sd}/{job}_{sd}_model.cif"
    if os.path.exists(cif):
        p, l = [], []
        for line in open(cif):
            if line.startswith("ATOM"):
                p.append(float(line.split()[14]))
            elif line.startswith("HETATM"):
                l.append(float(line.split()[14]))
        if p:
            out["plddt_prot"] = round(float(np.mean(p)), 1)
        if l:
            out["plddt_lig"] = round(float(np.mean(l)), 1)
            out["plddt_lig_min"] = round(float(np.min(l)), 1)
    return out


if __name__ == "__main__":
    jobs = sorted(os.path.basename(f)[:-4]
                  for f in glob.glob(f"{SITE}/pockets/*.pdb"))
    meta, contacts, miss = {}, {}, []
    for job in jobs:
        r = rank0(job)
        if not r:
            miss.append(job); continue
        meta[job] = r
        f = f"{AF3}/analysis/contacts/{job}__{r['sample']}.csv"
        if not os.path.exists(f):
            miss.append(job); continue
        rows = []
        for x in csv.DictReader(open(f)):
            rows.append([x["interaction_type"], x["protein_residue"],
                         int(x["protein_resnum"]), x["protein_chain"],
                         x["protein_atom"], x["ligand_atom"],
                         round(float(x["distance_A"]), 2)])
        contacts[job] = rows
    json.dump(dict(meta=meta, contacts=contacts,
                   cols=["itype", "resname", "resnum", "chain", "patom", "latom", "dist"]),
              open(f"{SITE}/rank0.json", "w"), separators=(",", ":"))
    sz = os.path.getsize(f"{SITE}/rank0.json") / 1024
    print(f"wrote site/rank0.json  ({sz:.0f} KB)")
    print(f"  {len(meta)} jobs, {sum(len(v) for v in contacts.values())} Rank 0 contact rows")
    if miss:
        print(f"  !! {len(miss)} jobs without a Rank 0 CSV: {miss[:4]}")
    which = collections.Counter(m["sample"] for m in meta.values())
    print(f"\n  which sample AF3 ranked first: {dict(sorted(which.items()))}")
    rs = [m["rank_score"] for m in meta.values()]
    mg = [m["margin"] for m in meta.values() if m.get("margin") is not None]
    print(f"  ranking score: mean {np.mean(rs):.3f}, range {min(rs):.3f}-{max(rs):.3f}")
    print(f"  margin over Rank 1: mean {np.mean(mg):.4f}, "
          f"under 0.01 in {sum(1 for x in mg if x < 0.01)} of {len(mg)} jobs")
