#!/usr/bin/env python3
"""Per job, the pLDDT of the model itself -- not of whatever the viewer cropped.

The viewer was averaging the B-factors of the atoms it had loaded, so in pocket
mode the "protein" figure was really the 8 A crop and it changed when you
switched to the raw tetramer. These are the model's own numbers, computed once
over the whole top-ranked structure, so they are the same whichever view is open.

Reported per job: the protein mean over all four chains, the ligand mean, the
ligand's range, and how the ligand's atoms fall into AlphaFold's confidence
bands -- the ligand is the number that matters here, and it is routinely far
lower than the protein's.
"""
import gzip, glob, json, os
import numpy as np

SITE = "/home/adsiordia/BSH-Model/site"
out = {}
for f in sorted(glob.glob(f"{SITE}/raw/*.pdb.gz")):
    job = os.path.basename(f)[:-7]
    prot, lig = [], []
    for l in gzip.open(f, "rt"):
        if l.startswith("ATOM"):
            prot.append(float(l[60:66]))
        elif l.startswith("HETATM"):
            lig.append(float(l[60:66]))
    if not prot:
        continue
    p, g = np.array(prot), np.array(lig) if lig else np.array([])
    band = lambda v: ["very low", "low", "confident", "very high"][
        int(v >= 50) + int(v >= 70) + int(v >= 90)]
    out[job] = dict(
        prot=round(float(p.mean()), 1), prot_n=int(p.size),
        lig=round(float(g.mean()), 1) if g.size else None,
        lig_n=int(g.size),
        lig_min=round(float(g.min()), 1) if g.size else None,
        lig_max=round(float(g.max()), 1) if g.size else None,
        lig_band=band(float(g.mean())) if g.size else None)

json.dump(out, open(f"{SITE}/plddt.json", "w"), separators=(",", ":"))
v = [o["lig"] for o in out.values() if o["lig"] is not None]
w = [o["prot"] for o in out.values()]
print(f"wrote site/plddt.json  ({os.path.getsize(f'{SITE}/plddt.json')/1024:.0f} KB, {len(out)} jobs)")
print(f"  protein pLDDT: mean {np.mean(w):.1f}, range {min(w):.1f}-{max(w):.1f}")
print(f"  ligand  pLDDT: mean {np.mean(v):.1f}, range {min(v):.1f}-{max(v):.1f}")
import collections
b = collections.Counter(o["lig_band"] for o in out.values())
print(f"  ligand confidence bands across the panel: {dict(b)}")
