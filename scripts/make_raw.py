#!/usr/bin/env python3
"""Write the whole raw tetramer per run, gzipped, for the web 3D viewer.

The pocket PDBs in make_pockets.py crop to 8 A around the ligand, which is what
the viewer needs for speed but hides the quaternary context -- you cannot see
that the ligand sits in one of four sites, nor that the residue reaching into it
belongs to a different chain. This writes the uncropped model instead: all four
protein chains and the ligand, straight from the top-ranked AlphaFold 3 output,
no cropping, no superposition.

~822 KB each as text, ~200 KB gzipped, so they are fetched on demand rather than
with the page. The B-factor column carries pLDDT, as in the pocket files.
"""
import warnings, os, sys, gzip, glob
warnings.filterwarnings("ignore")
from Bio.PDB import MMCIFParser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_pockets import _rec            # same fixed-column writer, same bug fixes

ROOT = "/home/adsiordia/AF3/BSH_AF3"
OUT = f"{ROOT}/analysis/raw"
p = MMCIFParser(QUIET=True)


def raw_pdb(cif):
    s = p.get_structure("x", cif)[0]
    lines, n = [], 0
    for c in s:                          # every protein chain, every residue
        if c.id == "L":
            continue
        for r in c:
            if r.id[0] != " ":
                continue
            for a in r:
                n += 1
                lines.append(_rec("ATOM  ", n, a, r.resname, c.id, r.id[1]))
    for c in s:                          # ligand last, as LIG (the field is 3 chars)
        if c.id != "L":
            continue
        for r in c:
            for a in r:
                n += 1
                lines.append(_rec("HETATM", n, a, "LIG", "L", 1))
    return ("\n".join(lines) + "\nEND\n") if n else None


def find_cif(job):
    for d in ("out_panel", "glytau/out", "round2/out"):
        f = f"{ROOT}/{d}/{job}/{job}_model.cif"
        if os.path.exists(f):
            return f
    return None


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    # mirror exactly the job set the site already ships pockets for
    jobs = sorted(os.path.basename(f)[:-4]
                  for f in glob.glob("/home/adsiordia/BSH-Model/site/pockets/*.pdb"))
    done = skip = 0
    for j in jobs:
        dst = f"{OUT}/{j}.pdb.gz"
        if os.path.exists(dst):
            skip += 1
            continue
        cif = find_cif(j)
        if not cif:
            print(f"  no cif: {j}")
            continue
        txt = raw_pdb(cif)
        if not txt:
            print(f"  no atoms: {j}")
            continue
        with gzip.open(dst, "wt", compresslevel=9) as fh:
            fh.write(txt)
        done += 1
        if done % 50 == 0:
            print(f"  {done} written", flush=True)
    print(f"done: {done} written, {skip} already present, {len(jobs)} jobs total")
