#!/usr/bin/env python3
"""Generate the BSH x amine AF3 panel: 34 proteins x 13 conjugated products, tetramers.

One directory per protein so AF3 computes each MSA once and reuses it for all
13 ligands in that batch (measured: 900 s for the first job, 0.05 s thereafter).

Job names use [A-Za-z0-9_] only.  AF3 strips other characters from the `name`
field when it creates the output directory -- the existing runs show
'C0ERS1_full_mono_GCA+Asn' landing in 'out/C0ERS1_full_mono_GCAAsn/' -- which
would silently break any mapping from input file to output folder.
"""
import json, csv, pathlib, sys

import sys as _sys, pathlib as _pl
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
from _paths import AF3_ROOT, need_af3
ROOT = AF3_ROOT
HERE = ROOT / "panel"
ARCH = "tet"
SEED = [1]
need_af3("the 675 predictions in out_panel/")

# Filesystem-safe token per amine.  Spelled out rather than abbreviated: these
# strings end up in every output path, CSV row and figure title.
# 'Dopa' is deliberately avoided -- L-DOPA is a different molecule.
TOKEN = {
    "asparagine": "Asn", "GABA": "GABA", "phenylalanine": "Phe",
    "dopamine": "Dopamine", "2_aminophenol": "2Aminophenol",
    "2,3_diaminopropionic acid": "Dap", "methionine": "Met", "serine": "Ser",
    "glutamine": "Gln", "3_methoxytyramine": "3MeOTyramine",
    "tyramine": "Tyramine", "tryptamine": "Tryptamine", "threonine": "Thr",
}

def load_fasta(path, key="after_first_underscore"):
    seqs, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:]
            if key == "after_first_underscore":
                code = name.split("_", 1)[1] if "_" in name else name
            else:                                  # trailing accession
                code = name.split("_")[-1]
            seqs[code] = ""
        elif name:
            seqs[code] += line
    return seqs


def mature_sequences():
    """Constructs that begin at the catalytic Cys.

    BSH is an Ntn-hydrolase: the catalytic Cys must be the N-terminal residue,
    because its free alpha-amino group is part of the catalytic machinery.

      * 22 panel proteins carry a signal peptide -> use the collaborator's
        pre-trimmed sequences verbatim.
      * the remaining 12 have no signal peptide and start M-C-T... -> strip
        Met1, as ligands.txt specifies for A0A0D8IVP2.

    A0A414Q275 is left exactly as the collaborator supplied it.  Its trimmed
    sequence begins 'TGITIKT', i.e. one residue past the Cys in the full
    sequence ('...ALAC|TGITIKT'); this is deliberate, not an oversight.
    """
    trimmed = load_fasta(ROOT / "proteins without signal peptides trimmed (1).txt",
                         key="trailing")
    raw = load_fasta(ROOT / "Seqs_list_total.fasta")
    out, source = {}, {}
    for code, seq in raw.items():
        if code in trimmed:
            out[code] = trimmed[code]
            source[code] = "signal_peptide_trimmed"
        elif seq.startswith("MC"):
            out[code] = seq[1:]
            source[code] = "met1_stripped"
        else:
            out[code] = seq
            source[code] = "raw_unchanged"
    return out, source

def main():
    panel = json.load(open(HERE / "panel.json"))
    conj = json.load(open(HERE / "conjugates.json"))
    picks = json.load(open(HERE / "protein_picks.json"))
    pct = json.load(open(HERE / "pct_tri.json"))
    seqs, seqsrc = mature_sequences()

    stratum_of = {c: s for s, v in picks.items() for c in v}
    proteins = [c for v in picks.values() for c in v] + ["C0ERS1", "A0A0D8IVP2"]
    stratum_of.setdefault("C0ERS1", "existing")
    stratum_of.setdefault("A0A0D8IVP2", "existing")

    # activity label: does this protein make this conjugate on the Tri core?
    import collections
    rows = list(csv.reader(open(ROOT / "Trimmed_remove_proteomics (4).csv")))
    header, data = rows[0], rows[1:]
    tri = collections.defaultdict(list)
    for j, c in enumerate(header[3:]):
        p = c.split("_")
        if p[0] == "Tri":
            tri["_".join(t for t in p[1:-1] if not t.startswith("M+"))].append(j + 3)
    active = collections.defaultdict(dict)
    for x in data:
        for a, js in tri.items():
            v = max((float(x[j]) for j in js if x[j] not in ("", "0")), default=0.0)
            active[x[1]][a] = max(active[x[1]].get(a, 0.0), v)

    jobs, missing = [], []
    for code in proteins:
        if code not in seqs:
            missing.append(code); continue
        seq = seqs[code]
        d = HERE / "jsons" / code
        d.mkdir(parents=True, exist_ok=True)
        for amine in panel:
            tok = TOKEN[amine]
            name = f"{code}_{ARCH}_{tok}"
            assert name.replace("_", "").isalnum(), f"unsafe job name: {name}"
            spec = {
                "name": name,
                "modelSeeds": SEED,
                "sequences": [
                    {"protein": {"id": ["A", "B", "C", "D"], "sequence": seq}},
                    {"ligand": {"id": "L", "smiles": conj[amine]["smiles"]}},
                ],
                "dialect": "alphafold3",
                "version": 1,
            }
            (d / f"{name}.json").write_text(json.dumps(spec, indent=2))
            intensity = active.get(code, {}).get(amine, 0.0)
            jobs.append({
                "job": name, "protein": code, "stratum": stratum_of[code],
                "amine": amine, "amine_token": tok,
                "formula": conj[amine]["formula"], "product_mw": conj[amine]["mw"],
                "smiles": conj[amine]["smiles"],
                "tri_prevalence_pct": round(pct.get(amine, 0.0), 1),
                "active": int(intensity > 0), "intensity": intensity,
                "ligand_resname": "LIG_L", "construct": seqsrc[code],
                "seq_len": len(seq), "starts_with": seq[:8],
                "input_json": str(d / f"{name}.json"),
                "expected_outdir": str(ROOT / "out_panel" / name),
            })

    with open(HERE / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(jobs[0]))
        w.writeheader(); w.writerows(jobs)
    (HERE / "proteins.txt").write_text("\n".join(
        c for c in proteins if c in seqs) + "\n")

    print(f"proteins: {len(proteins) - len(missing)}  amines: {len(panel)}  jobs: {len(jobs)}")
    if missing:
        print(f"SKIPPED (no sequence in the FASTA): {missing}")
    print(f"manifest: {HERE/'manifest.csv'}")

if __name__ == "__main__":
    main()
