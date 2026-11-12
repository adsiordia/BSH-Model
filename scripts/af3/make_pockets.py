#!/usr/bin/env python3
"""Write a pocket-only PDB per run for the web 3D viewer.

The full tetramer is ~0.94 MB; everything within 8 A of the ligand is ~46 KB,
which is what the viewer actually needs. The B-factor column carries AF3's
pLDDT, so the viewer can colour by confidence.
"""
import warnings, os, sys, glob, csv
import numpy as np
warnings.filterwarnings("ignore")
from Bio.PDB import MMCIFParser

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from _paths import AF3_ROOT, need_af3
ROOT = str(AF3_ROOT)
OUT = f"{ROOT}/analysis/pockets"
CUT = 8.0
need_af3("the 675 predictions in out_panel/")
p = MMCIFParser(QUIET=True)


def _rec(kind, serial, atom, resname, chain, resseq):
    """One PDB ATOM/HETATM record, correctly columned.

    The fields are fixed-width and unforgiving: atom name 13-16 (with the
    element right-justified in 13-14, so a one-letter element gets a leading
    space), resName 18-20, chainID 22, resSeq 23-26, x/y/z 31-54, occupancy
    55-60, tempFactor 61-66, element 77-78. Getting this wrong by one column
    silently drops most atoms in any strict parser.
    """
    el = atom.element.strip()
    nm = atom.name.strip()
    name = f"{nm:<4}" if len(el) == 2 or len(nm) == 4 else f" {nm:<3}"
    return (f"{kind}{serial:>5} {name}"
            f" {resname[:3]:>3} {chain}{resseq:>4}    "
            f"{atom.coord[0]:8.3f}{atom.coord[1]:8.3f}{atom.coord[2]:8.3f}"
            f"  1.00{atom.bfactor:6.2f}          {el:>2}")


def pocket_pdb(cif):
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
    for a in lig:                      # ligand last, as LIG (the field is 3 chars)
        n += 1
        lines.append(_rec("HETATM", n, a, "LIG", "L", 1))
    return "\n".join(lines) + "\nEND\n"


def main():
    os.makedirs(OUT, exist_ok=True)
    keep = {l.strip() for l in open(f"{ROOT}/analysis/keep_proteins.txt") if l.strip()}
    jobs = []
    for base in ("out_panel", "round2/out", "glytau/out"):
        for d in sorted(glob.glob(f"{ROOT}/{base}/*/")):
            j = os.path.basename(d.rstrip("/"))
            if "_tet_" in j and j.split("_tet_")[0] in keep:
                jobs.append((base, j))
    done = skip = 0
    for base, j in jobs:
        out = f"{OUT}/{j}.pdb"
        if os.path.exists(out) and os.path.getsize(out) > 0:
            skip += 1
            continue
        cif = f"{ROOT}/{base}/{j}/{j}_model.cif"
        if not os.path.exists(cif):
            continue
        txt = pocket_pdb(cif)
        if txt:
            open(out, "w").write(txt)
            done += 1
    print(f"wrote {done}, already present {skip}, total {len(jobs)}")


if __name__ == "__main__":
    main()
