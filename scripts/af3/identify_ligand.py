#!/usr/bin/env python3
"""Recover which amine a run used, from the structure file alone.

Every ligand in the panel is named LIG_L so that `pandamap --ligand LIG_L`
works identically across all 442 runs.  The amine identity is NOT in the
residue name -- it lives in the job name, in manifest.csv, and in the SMILES
that AF3 records under _chem_comp.pdbx_smiles inside every model.cif.

    python3 identify_ligand.py ../out_panel/*/*_model.cif
"""
import csv, json, pathlib, re, sys
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")   # we probe lines that are not SMILES

HERE = pathlib.Path(__file__).parent


def cif_ligand_smiles(path):
    """Return {resname: smiles} from the _chem_comp loop of an AF3 mmCIF.

    AF3 writes one _chem_comp row per component; for a user ligand the
    formula/weight columns are '?' and the SMILES sits in pdbx_smiles:

        ?  ?  LIG_L  ?  ?  O[C@@H]1CC[C@]2(C)...  ?  non-polymer

    Only rows inside that loop are considered, so coordinate lines that also
    mention LIG_ are never probed.
    """
    out, in_loop = {}, False
    for line in open(path):
        if line.startswith("_chem_comp."):
            in_loop = True
            continue
        if in_loop:
            if line.startswith(("#", "loop_", "_")) or not line.strip():
                if out:
                    break
                in_loop = line.startswith("_chem_comp.")
                continue
            if "LIG_" not in line:
                continue
            fields = line.split()
            resname = next((f for f in fields if f.startswith("LIG_")), None)
            if resname is None:
                continue
            for f in fields:
                if len(f) > 10 and Chem.MolFromSmiles(f) is not None:
                    out[resname] = f
                    break
    return out


def lookup_table():
    """canonical SMILES -> amine name, from the panel's conjugates.json."""
    conj = json.load(open(HERE / "conjugates.json"))
    return {Chem.CanonSmiles(v["smiles"]): k for k, v in conj.items()}


def main(paths):
    table = lookup_table()
    manifest = {}
    mpath = HERE / "manifest.csv"
    if mpath.exists():
        manifest = {r["job"]: r for r in csv.DictReader(open(mpath))}
    print(f"{'file':<46}{'resname':<9}{'formula':<14}{'MW':>8}  amine (from SMILES) / manifest")
    for p in paths:
        p = pathlib.Path(p)
        job = p.name.replace("_model.cif", "")
        for resname, smi in sorted(cif_ligand_smiles(p).items()):
            mol = Chem.MolFromSmiles(smi)
            amine = table.get(Chem.CanonSmiles(smi), "UNKNOWN (not in this panel)")
            via_manifest = manifest.get(job, {}).get("amine", "-")
            agree = "" if via_manifest in ("-", amine) else "   <-- DISAGREES WITH MANIFEST"
            print(f"{p.name[:45]:<46}{resname:<9}"
                  f"{rdMolDescriptors.CalcMolFormula(mol):<14}{Descriptors.MolWt(mol):>8.2f}  "
                  f"{amine} / {via_manifest}{agree}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(__doc__); sys.exit(1)
    main(args)
