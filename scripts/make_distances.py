#!/usr/bin/env python3
"""Per job, PandaMap's own contacts for the model the viewer actually displays.

Distances on this site are PandaMap's, measured between the specific atom pair
that makes each interaction -- an Arg NH2 to a sulfonate oxygen for an ionic
contact, say. That is NOT the minimum heavy-atom separation between residue and
ligand, so it cannot be recomputed in the browser without contradicting every
table on the site.

PandaMap ran per sample, and the viewer shows the top-ranked model, which is
byte-identical to the highest-scoring sample's model (verified). So for each job
this picks the top sample out of ranking_scores.csv and keeps that sample's rows:
atom pair, distance and interaction type, all consistent with the geometry on
screen.

Contacts listed in the By-enzyme table but absent here are the ones seen only in
other samples -- the table is the union over five, the viewer is one model.
"""
import csv, glob, json, os, collections

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import AF3_ROOT, need_af3
ROOT = str(AF3_ROOT)
CSVD = f"{ROOT}/analysis/contacts"
need_af3("the 675 predictions in out_panel/")


def top_sample(job):
    for d in ("out_panel", "glytau/out", "round2/out"):
        f = f"{ROOT}/{d}/{job}/{job}_ranking_scores.csv"
        if os.path.exists(f):
            rows = list(csv.DictReader(open(f)))
            if not rows:
                return None
            best = max(rows, key=lambda r: float(r["ranking_score"]))
            return f"seed-{best['seed']}_sample-{best['sample']}"
    return None


if __name__ == "__main__":
    jobs = sorted(os.path.basename(f)[:-4]
                  for f in glob.glob(f"{_SITE}/pockets/*.pdb"))
    out, miss, ntot = {}, 0, 0
    for job in jobs:
        samp = top_sample(job)
        if not samp:
            miss += 1
            continue
        f = f"{CSVD}/{job}__{samp}.csv"
        if not os.path.exists(f):
            miss += 1
            continue
        rows = []
        for r in csv.DictReader(open(f)):
            rows.append([r["protein_residue"], int(r["protein_resnum"]),
                         r["protein_chain"], r["protein_atom"], r["ligand_atom"],
                         round(float(r["distance_A"]), 2), r["interaction_type"]])
        out[job] = rows
        ntot += len(rows)
    dst = f"{_SITE}/distances.json"
    json.dump(out, open(dst, "w"), separators=(",", ":"))
    print(f"wrote {dst}  ({os.path.getsize(dst)/1024:.0f} KB)")
    print(f"  {len(out)} jobs, {ntot} contacts from the displayed model, {miss} missing")
