#!/usr/bin/env python3
"""The conjugated amines drawn as structures, as one strip aligned to a grid.

The amines are not typed out here. Each one is cut from the conjugate SMILES
AlphaFold was actually given (the ligand block of every *_tet_*.json), by
breaking the single amide whose acyl half carries the four-ring steroid. The
nitrogen freed by that cut is the one that bonded to the bile acid, so it is
known rather than guessed -- which matters for ornithine, Dap and the diamines,
where a second free amine would otherwise be indistinguishable from it.

Why a strip and not 25 separate annotations: the columns are the only thing that
has to line up, and one image on an axes sharing the heatmap's x-limits lines up
by construction. Tiles are square and equal width, so tile i sits under column i.

Each molecule is drawn into a canvas narrower than its tile and pasted centred,
so GUTTER of the tile width is empty whatever the molecule's shape and no two
structures can touch. Cadaverine and putrescine are the test: drawn to the tile
edge they run into each other. Atom labels are pinned to one size rather than
scaled with the molecule, so a label is equally readable over a ring and over a
six-carbon chain.

Fixed bond length was tried first, which keeps the amines comparable in size,
but at 25 columns a tile is about 0.4 inch wide and glycine then renders as a
speck -- the structures stop being readable, which is the only reason they are
here. Relative size is already given by the names; legibility is not.

Atoms are monochrome so the only hue in the figure is the heatmap's magnitude
ramp; the conjugating nitrogen carries a halo instead.
"""
import glob
import io
import json
import os

from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
from _paths import AF3_ROOT
AF3 = str(AF3_ROOT)
HALO = (0.984, 0.851, 0.784)          # tint of categorical orange #eb6834
# TILE is the strip's own resolution per column. The strip is drawn once at
# this size and matplotlib scales it down to whatever the column is worth on the
# page, so TILE only has to be comfortably larger than that -- at 25 columns
# over ~10 inches a column is ~125 px at 300 dpi, and 560 leaves 4x in hand.
# Everything else is a fraction of TILE*SS so raising it rescales cleanly.
TILE, PAD, GUTTER, SS = 560, .10, .10, 2
FONT = round(.084 * TILE * SS)
LW = round(.0125 * TILE * SS)
AMIDE = Chem.MolFromSmarts("[CX3](=O)[NX3]")


def conjugates():
    """token -> the conjugate SMILES AlphaFold was given, from the job inputs."""
    out = {}
    for f in (glob.glob(f"{AF3}/*/jsons/**/*_tet_*.json", recursive=True)
              + glob.glob(f"{AF3}/*/jsons/*_tet_*.json")):
        d = json.load(open(f))
        tok = (d.get("name") or os.path.basename(f)[:-5]).split("_tet_")[-1]
        for s in d.get("sequences", []):
            if "ligand" in s and "smiles" in s["ligand"]:
                prev = out.setdefault(tok, s["ligand"]["smiles"])
                if prev != s["ligand"]["smiles"]:
                    raise ValueError(f"{tok}: two different ligands")
    return out


def amine(conj_smiles):
    """Cut the steroid off at the amide; return the amine, conjugating N mapped 1."""
    base = Chem.MolFromSmiles(conj_smiles)
    for c, _o, n in base.GetSubstructMatches(AMIDE):
        m = Chem.RWMol(base)
        m.GetAtomWithIdx(n).SetAtomMapNum(1)
        m.RemoveBond(c, n)
        frags = Chem.GetMolFrags(m.GetMol(), asMols=True, sanitizeFrags=False)
        if len(frags) != 2:
            continue
        for x in frags:
            Chem.FastFindRings(x)
        hit = [x for x in frags if any(a.GetAtomMapNum() == 1 for a in x.GetAtoms())]
        rest = [x for x in frags if x is not hit[0]]
        if not hit or rdMolDescriptors.CalcNumRings(rest[0]) < 4:
            continue                      # an amide inside the amine, not the link
        a = hit[0]
        Chem.SanitizeMol(a)
        return a
    raise ValueError(f"no steroid amide in {conj_smiles}")


def tile(mol):
    m = Chem.Mol(mol)
    idx = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomMapNum() == 1]
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)                # or RDKit prints ":1" on the nitrogen
    AllChem.Compute2DCoords(m)
    box = round(TILE * (1 - GUTTER)), TILE          # the drawable part of a tile
    d = rdMolDraw2D.MolDraw2DCairo(box[0] * SS, box[1] * SS)
    o = d.drawOptions()
    o.useBWAtomPalette()
    o.clearBackground = False
    o.padding = PAD
    o.bondLineWidth = LW
    o.highlightRadius = .42
    o.minFontSize = FONT
    o.maxFontSize = FONT
    o.setHighlightColour(HALO)
    rdMolDraw2D.PrepareAndDrawMolecule(d, m, highlightAtoms=idx)
    d.FinishDrawing()

    im = Image.open(io.BytesIO(d.GetDrawingText())).convert("RGBA")
    im = im.resize(box, Image.LANCZOS)              # supersampled down, so sharp
    out = Image.new("RGBA", (TILE, TILE), (0, 0, 0, 0))
    out.paste(im, ((TILE - box[0]) // 2, 0))
    return out


def strip(tokens, out):
    """One transparent PNG, len(tokens) square tiles wide."""
    conj = conjugates()
    im = Image.new("RGBA", (TILE * len(tokens), TILE), (0, 0, 0, 0))
    for i, t in enumerate(tokens):
        im.paste(tile(amine(conj[t])), (i * TILE, 0))
    im.save(out)
    return out


if __name__ == "__main__":
    conj = conjugates()
    toks = sorted(conj)
    print(f"{len(toks)} conjugates found in the AlphaFold job inputs")
    for t in toks:
        a = amine(conj[t])
        n = sum(1 for x in a.GetAtoms() if x.GetAtomMapNum() == 1)
        free = sum(1 for x in a.GetAtoms()
                   if x.GetSymbol() == "N" and x.GetTotalNumHs() and x.GetAtomMapNum() == 0)
        print(f"  {t:<14} {Chem.MolToSmiles(a):<42} linked N {n}"
              + (f", {free} other free amine(s)" if free else ""))
    out = strip(toks, f"{_REPO}/figures/amine_strip.png")
    print(f"wrote {out}")
