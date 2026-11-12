#!/usr/bin/env python3
"""Reusable trend analysis for the BSH x amine panel.

Everything is keyed on a COMMON RESIDUE NUMBERING (pairwise alignment to
C0ERS1's mature sequence), so positions are comparable across proteins whose
own numbering differs by tens of residues.

Each contact is tagged:
  part  = 'core'   (the cholic acid steroid, identical in every conjugate)
          'moiety' (the amine, the only part that varies)
  chain = 'own'       (the subunit that supplies the catalytic Cys)
          'interface' (a neighbouring subunit -- the selectivity loop)

Values are CONTACT FREQUENCY over the 5 AF3 samples (0-5), not presence in a
single model, because ligand pLDDT in these tetramers is only moderate.

Interaction types kept: hydrogen_bonds, ionic, attractive_charge, hydrophobic,
repulsion.  All pi-types are dropped: PandaMap calls any ring carbon aromatic,
so the saturated steroid rings generate large numbers of spurious pi contacts.

Usage:
    from bsh_trends import Panel
    P = Panel()
    P.position_profile(206)            # who contacts position 206, by amine
    P.residue_at(206)                  # residue identity per protein
    P.contact_matrix(part='moiety')    # positions x amines frequency matrix
    P.compare_groups(206, P.motif_groups())
"""
import pickle, csv, collections, json
import numpy as np

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from _paths import AF3_ROOT
ROOT = str(AF3_ROOT)
AN = f"{ROOT}/analysis"


class Panel:
    def __init__(self):
        self.cols, self.rows = pickle.load(open(f"{AN}/all_contacts.pkl", "rb"))
        self.I = {c: i for i, c in enumerate(self.cols)}
        self.motif = pickle.load(open(f"{AN}/motif.pkl", "rb"))
        self.num = pickle.load(open(f"{AN}/numbering.pkl", "rb"))
        self.man = {r["job"]: r
                    for r in csv.DictReader(open(f"{ROOT}/panel/manifest.csv"))}
        self.proteins = sorted({r[self.I["protein"]] for r in self.rows})
        self.amines = sorted({r[self.I["token"]] for r in self.rows})
        self._inv = {p: {v: k for k, v in m.items()}
                     for p, m in self.num["to_ref"].items()}
        self._seq = {}

    # ---------- sequence helpers ----------
    def seq(self, p):
        if p not in self._seq:
            j = json.load(open(f"{ROOT}/panel/jsons/{p}/{p}_tet_Asn.json"))
            self._seq[p] = [s["protein"]["sequence"]
                            for s in j["sequences"] if "protein" in s][0]
        return self._seq[p]

    def residue_at(self, pos):
        """{protein: 'R206'} in the protein's OWN numbering, for a reference position."""
        out = {}
        for p in self.proteins:
            j = self._inv.get(p, {}).get(pos)
            out[p] = f"{self.seq(p)[j-1]}{j}" if j else None
        return out

    # ---------- grouping ----------
    def motif_groups(self):
        """Foley et al. selectivity-loop classes from reference positions 205-207."""
        g = collections.defaultdict(list)
        for p, m in self.motif.items():
            if m[1] == "R":                         k = "S-R-X (Arg centre)"
            elif m[0] == "G" and m[2] == "G":       k = "G-X-G"
            elif "-" in m:                          k = "no loop"
            else:                                    k = "Other"
            g[k].append(p)
        return dict(g)

    # ---------- core queries ----------
    def freq(self, pos=None, part=None, chain=None, token=None, protein=None):
        """Contact frequency (0-5) per job, filtered."""
        seen = collections.defaultdict(set)
        for r in self.rows:
            if pos is not None and r[self.I["pos"]] != pos: continue
            if part and r[self.I["part"]] != part: continue
            if token and r[self.I["token"]] != token: continue
            if protein and r[self.I["protein"]] != protein: continue
            cls = "interface" if r[self.I["chain"]] != "A" else "own"
            if chain and cls != chain: continue
            seen[r[self.I["job"]]].add(r[self.I["sample"]])
        return {j: len(v) for j, v in seen.items()}

    def position_profile(self, pos, part="moiety", chain=None):
        """mean 0-5 frequency at `pos`, per amine."""
        f = self.freq(pos=pos, part=part, chain=chain)
        out = {}
        for t in self.amines:
            vals = [f.get(f"{p}_tet_{t}", 0) for p in self.proteins]
            out[t] = float(np.mean(vals))
        return out

    def contact_matrix(self, part="moiety", chain=None, min_var=0.0):
        """positions x amines matrix of mean frequency; rows with low variance dropped."""
        positions = sorted({r[self.I["pos"]] for r in self.rows
                            if (not part or r[self.I["part"]] == part)})
        M, keep = [], []
        for pos in positions:
            row = [self.position_profile(pos, part, chain)[t] for t in self.amines]
            if np.std(row) >= min_var and np.sum(row) > 0:
                M.append(row); keep.append(pos)
        return np.array(M), keep, self.amines

    def compare_groups(self, pos, groups, part="moiety", chain="interface",
                       split=None, n_perm=20000, seed=0):
        """Per-group paired comparison of frequency at `pos` between two amine sets."""
        if split is None:
            split = self.cooh_split()
        f = self.freq(pos=pos, part=part, chain=chain)
        rng = np.random.default_rng(seed)
        out = {}
        for g, ps in groups.items():
            a = [np.mean([f.get(f"{p}_tet_{t}", 0) for t in split[True]]) for p in ps]
            b = [np.mean([f.get(f"{p}_tet_{t}", 0) for t in split[False]]) for p in ps]
            d = np.array(a) - np.array(b)
            if len(d) < 2:
                out[g] = (np.mean(a), np.mean(b), float(d.mean()), np.nan, len(ps)); continue
            perm = [np.mean(d * rng.choice([-1, 1], len(d))) for _ in range(n_perm)]
            p_ = (np.sum(np.abs(perm) >= abs(d.mean())) + 1) / (n_perm + 1)
            out[g] = (float(np.mean(a)), float(np.mean(b)), float(d.mean()), float(p_), len(ps))
        return out

    def cooh_split(self):
        """{True: [amines with a carboxylate], False: [without]}"""
        import sys
        sys.path.insert(0, f"{ROOT}/panel")
        from amines import SMILES
        from rdkit import Chem
        TOK = json.load(open(f"{AN}/token_map.json"))
        patt = Chem.MolFromSmarts("[CX3](=O)[OX2H1,OX1-]")
        d = {True: [], False: []}
        for amine, tok in TOK.items():
            has = Chem.MolFromSmiles(SMILES[amine]).HasSubstructMatch(patt)
            d[has].append(tok)
        return d


if __name__ == "__main__":
    P = Panel()
    print(f"{len(P.proteins)} proteins x {len(P.amines)} amines")
    print("\nmotif groups:", {k: len(v) for k, v in P.motif_groups().items()})
    print("\nposition 206, interface, amine moiety -- mean frequency per amine:")
    for t, v in sorted(P.position_profile(206, "moiety", "interface").items(),
                       key=lambda kv: -kv[1]):
        print(f"   {t:<14}{v:.2f}")
    print("\ncarboxylate test at position 206:")
    for g, (a, b, d, p_, n) in P.compare_groups(206, P.motif_groups()).items():
        print(f"   {g:<22} n={n:<3} COOH {a:.2f}  nonCOOH {b:.2f}  diff {d:+.2f}  p={p_:.4f}")
