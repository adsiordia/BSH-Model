#!/usr/bin/env python3
"""Write site/xtal/<PDB>.pdb.gz -- the crystal tetramers the Crystal check viewer fetches.

site/xtal/ is gitignored, so these have to be regenerable. .gitignore pointed at
make_validation.py for that, but nothing there writes them; this script does.

Conventions the shipped files follow, and which 3Dmol and the tab's residue
lookup both depend on:

  * waters dropped -- they are a third of the atoms and nothing references them
  * the catalytic cysteine travels as ATOM, not HETATM, even when it is the
    oxidised OCS. Bio.PDB would write it HETATM, which would hide it from the
    viewer's cartoon, so those records are rewritten.
  * every chain of the assembly is kept; the tab's `copies` selector is keyed by
    the ligand's chain

2BJF is NOT handled here. Only one chain is deposited and its tetramer has to be
rebuilt from the four crystal symmetry operators first (refstruct/2BJF_tet.cif);
run this on that file rather than on the deposit.
"""
import gzip
import io
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
from Bio.PDB import MMCIFParser, PDBIO, Select

REF = Path("/home/adsiordia/AF3/BSH_AF3/refstruct")
SITE = Path(__file__).resolve().parent.parent / "site/xtal"
# the catalytic cysteine, oxidised in both of these deposits
AS_ATOM = {"OCS"}


class NoWater(Select):
    def accept_residue(self, r):
        return 0 if r.resname == "HOH" else 1


def build(pdb, cif=None):
    src = REF / (cif or f"{pdb}.cif")
    st = MMCIFParser(QUIET=True).get_structure(pdb, src)
    io_ = PDBIO()
    io_.set_structure(st)
    buf = io.StringIO()
    io_.save(buf, select=NoWater())
    lines = []
    for l in buf.getvalue().splitlines(True):
        if l.startswith("HETATM") and l[17:20].strip() in AS_ATOM:
            l = "ATOM  " + l[6:]
        lines.append(l)
    SITE.mkdir(parents=True, exist_ok=True)
    dest = SITE / f"{pdb}.pdb.gz"
    text = "".join(lines)
    # mtime=0 and no embedded filename, so the bytes are a function of the
    # content alone and the shipped file can be checked against a rebuild
    for out in (dest, REF / f"{pdb}.pdb.gz"):
        with open(out, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
                f.write(text.encode())
    n_at = sum(1 for l in lines if l.startswith(("ATOM", "HETATM")))
    chains = "".join(sorted({l[21] for l in lines if l.startswith(("ATOM", "HETATM"))}))
    het = sorted({l[17:20].strip() for l in lines if l.startswith("HETATM")})
    print(f"{pdb}: {dest.stat().st_size:,} B  {n_at:,} atoms  chains {chains}  ligands {het}")


if __name__ == "__main__":
    for pdb in (sys.argv[1:] or ["7SVG"]):
        build(pdb)
