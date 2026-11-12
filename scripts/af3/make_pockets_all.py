#!/usr/bin/env python3
"""Pocket PDBs for all five AlphaFold samples, not just the top-ranked one.

make_pockets.py wrote one file per job, from Rank 0. But the five samples
genuinely differ, and that variation is the thing the 3-of-5 threshold measures:
position 219's activity signal exists only because some samples place a
pi-stacking contact there and others do not. Every number on the site stays on
the >=3 of 5 consensus; these files just let the viewer show any one of the five
so the structural variation behind that consensus can be seen rather than taken
on trust.

Same 8 A crop, same fixed-column writer, pLDDT in the B-factor column. Named
{job}__seed-S_sample-N.pdb to match the PandaMap CSVs exactly.
"""
import warnings, os, sys, glob
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_pockets import _rec, CUT
from Bio.PDB import MMCIFParser

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from _paths import AF3_ROOT, SITE as _SITE, need_af3
need_af3("the 675 predictions in out_panel/")

ROOT = str(AF3_ROOT)
OUT = f"{ROOT}/analysis/pockets_all"
p = MMCIFParser(QUIET=True)


def pocket(cif):
    s = p.get_structure("x", cif)[0]
    lig = [a for c in s if c.id == "L" for r in c for a in r]
    if not lig:
        return None
    L = np.array([a.coord for a in lig])
    lines, n = [], 0
    for c in s:
        if c.id == "L":
            continue
        for r in c:
            if r.id[0] != " ":
                continue
            xyz = np.array([a.coord for a in r])
            if np.sqrt(((xyz[:, None, :] - L[None, :, :]) ** 2).sum(-1)).min() >= CUT:
                continue
            for a in r:
                n += 1
                lines.append(_rec("ATOM  ", n, a, r.resname, c.id, r.id[1]))
    for a in lig:
        n += 1
        lines.append(_rec("HETATM", n, a, "LIG", "L", 1))
    return "\n".join(lines) + "\nEND\n" if n else None


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    jobs = sorted(os.path.basename(f)[:-4]
                  for f in glob.glob(f"{_SITE}/pockets/*.pdb"))
    done = skip = fail = 0
    for job in jobs:
        jd = None
        for d in ("out_panel", "glytau/out", "round2/out"):
            if os.path.isdir(f"{ROOT}/{d}/{job}"):
                jd = f"{ROOT}/{d}/{job}"
                break
        if jd is None:
            continue
        for sd in sorted(glob.glob(f"{jd}/seed-*_sample-*")):
            samp = os.path.basename(sd)
            dst = f"{OUT}/{job}__{samp}.pdb"
            if os.path.exists(dst):
                skip += 1
                continue
            cif = f"{sd}/{job}_{samp}_model.cif"
            if not os.path.exists(cif):
                fail += 1
                continue
            txt = pocket(cif)
            if not txt:
                fail += 1
                continue
            open(dst, "w").write(txt)
            done += 1
            if done % 250 == 0:
                print(f"  {done} written", flush=True)
    print(f"done: {done} written, {skip} already there, {fail} failed")
