"""Every nitrogen environment in each amine, not just one label.

The old `cls` field gave each amine a single class, which misreads anything
with more than one kind of nitrogen -- arginine came out "primary", ignoring
its guanidino group entirely.
"""
from pathlib import Path
import json
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent

# order matters only for display; each is tested independently
PATTERNS = [
    ("guanidine",      "[NX3][CX3](=[NX2,NX3+])[NX3]"),
    ("urea",           "[NX3][CX3](=[OX1])[NX3]"),
    ("amide",          "[NX3;!$(N[CX3](=[OX1])[NX3])][CX3]=[OX1]"),
    ("aromatic amine", "[NX3;H1,H2;$(Nc)]"),
    ("imidazole",      "[nX2,nX3;r5;$(n1cncc1),$(n1ccnc1)]"),
    ("indole NH",      "[nX3H;r5;$(n1ccc2ccccc12),$(n1cc2ccccc2c1)]"),
    ("primary amine",  "[NX3;H2;!$(N[CX3]=[OX1]);!$(Nc);!$(N[CX3]=[NX2,NX3])]"),
    ("secondary amine","[NX3;H1;!$(N[CX3]=[OX1]);!$(Nc);!$(N[CX3]=[NX2,NX3]);!$([nX3H])]"),
    ("tertiary amine", "[NX3;H0;!$(N[CX3]=[OX1]);!$(Nc);!$(N[CX3]=[NX2,NX3]);!$([N+]);!a]"),
]
COMPILED = [(n, Chem.MolFromSmarts(s)) for n, s in PATTERNS]

# which environments can actually attack the C24 carbonyl
NUCLEOPHILIC = {"primary amine", "secondary amine", "aromatic amine"}


def classify(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    counts = {}
    for name, patt in COMPILED:
        n = len(m.GetSubstructMatches(patt))
        if n:
            counts[name] = n
    total_n = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "N")
    return dict(classes=counts,
                n_nitrogen=total_n,
                nucleophilic=sum(v for k, v in counts.items() if k in NUCLEOPHILIC))


if __name__ == "__main__":
    chem = json.loads((ROOT / "site/chem.json").read_text())
    out = {}
    for a, v in sorted(chem["amines"].items()):
        r = classify(v["smiles"])
        out[a] = r
        cls = ", ".join(f"{k}×{n}" if n > 1 else k for k, n in r["classes"].items())
        print(f"  {a:28s} old={v['cls']:10s} ->  {cls or '(none found)'}")
    (ROOT / "data/derived/nitrogen_classes.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote data/derived/nitrogen_classes.json")

    # Write the same three fields back into chem.json, which is what the amine
    # detail card renders. export_amine_chem.py does not produce them, so
    # without this step a chem.json rebuild silently drops the per-amine
    # nitrogen breakdown and the card falls back to the single `cls` label.
    cp = ROOT / "site/chem.json"
    for a, v in chem["amines"].items():
        r = out.get(a)
        if not r:
            continue
        v["n_classes"] = r["classes"]
        v["n_nitrogen"] = r["n_nitrogen"]
        v["label"] = ", ".join(f"{k} \u00d7{n}" if n > 1 else k
                               for k, n in r["classes"].items()) or "(none found)"
    cp.write_text(json.dumps(chem, separators=(",", ":")))
    print(f"patched site/chem.json with n_classes/n_nitrogen/label "
          f"for {sum(1 for a in chem['amines'] if 'n_classes' in chem['amines'][a])} amines")
