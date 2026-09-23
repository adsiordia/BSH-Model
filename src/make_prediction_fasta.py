"""Convert the two signal-peptide text files into proper FASTA.

Input (data/predictions_data/signaling_peptide/):
    SignalPeptides.txt                             full sequences
    proteins without signal peptides trimmed (1).txt   the same 66, N-terminus cut

Both are already FASTA-shaped but carry a .txt extension and inconsistent line
wrapping -- one record is a single 274-character line, the next is wrapped at 60.
This rewrites both at a uniform 60 columns, verifies the two files describe the
same 66 proteins, and checks that each trimmed sequence really is a suffix of its
full counterpart (which is what "signal peptide removed" has to mean).

Each FASTA keeps its source file's name, so the two stay easy to match up:

    SignalPeptides.txt                             -> SignalPeptides.fasta
    proteins without signal peptides trimmed (1).txt -> ...trimmed (1).fasta

A manifest alongside them gives both lengths and the piece that was cut.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/predictions_data/signaling_peptide"
OUT = SRC                       # keep the FASTA next to the .txt it came from
WRAP = 60
AA = set("ACDEFGHIKLMNPQRSTVWY")
FILES = {"full": "SignalPeptides.txt",
         "mature": "proteins without signal peptides trimmed (1).txt"}


def parse(path):
    recs, name, seq = [], None, []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                recs.append((name, "".join(seq)))
            name, seq = line[1:].strip(), []
        elif line:
            seq.append(line)
    if name is not None:
        recs.append((name, "".join(seq)))
    return recs


def write_fasta(recs, path):
    with open(path, "w") as f:
        for name, seq in recs:
            f.write(f">{name}\n")
            for i in range(0, len(seq), WRAP):
                f.write(seq[i:i + WRAP] + "\n")
    return path


def main():
    R = {k: parse(SRC / v) for k, v in FILES.items()}
    for k, recs in R.items():
        ids = [n for n, _ in recs]
        assert len(ids) == len(set(ids)), f"{k}: duplicate ids"
        assert all(s for _, s in recs), f"{k}: empty sequence"
        bad = {n: sorted(set(s) - AA) for n, s in recs if set(s) - AA}
        assert not bad, f"{k}: non-standard residues {bad}"

    full, mature = dict(R["full"]), dict(R["mature"])
    assert set(full) == set(mature), "the two files do not describe the same proteins"
    notsuffix = [n for n in full if not full[n].endswith(mature[n])]
    assert not notsuffix, f"mature is not a suffix of full for {notsuffix}"

    # same basename as the source, .fasta instead of .txt
    paths = {k: write_fasta(R[k], OUT / (Path(v).stem + ".fasta")) for k, v in FILES.items()}

    rows = [dict(id=n, full_length=len(full[n]), mature_length=len(mature[n]),
                 signal_peptide_length=len(full[n]) - len(mature[n]),
                 signal_peptide=full[n][:len(full[n]) - len(mature[n])])
            for n in sorted(full)]
    m = pd.DataFrame(rows).sort_values("signal_peptide_length")
    m.to_csv(OUT / "signal_peptide_manifest.csv", index=False)

    print(f"{len(rows)} proteins, ids identical in both files, "
          f"every mature sequence a suffix of its full sequence\n")
    for k, p in paths.items():
        L = [len(s) for _, s in R[k]]
        print(f"  {p.relative_to(ROOT)}")
        print(f"     {len(L)} records, {min(L)}-{max(L)} aa (median {sorted(L)[len(L)//2]})")
    print(f"\n  {(OUT/'signal_peptide_manifest.csv').relative_to(ROOT)}")
    print(f"     signal peptide: median {int(m.signal_peptide_length.median())} aa, "
          f"range {m.signal_peptide_length.min()}-{m.signal_peptide_length.max()}")
    odd = m[(m.signal_peptide_length < 10) | (m.signal_peptide_length > 50)]
    if len(odd):
        print(f"\n  {len(odd)} cuts fall outside the usual 15-40 aa range for a signal "
              f"peptide -- worth a look before trusting the mature sequence:")
        for _, r in odd.iterrows():
            print(f"     {r.id:38s} cut {r.signal_peptide_length:3d} aa   {r.signal_peptide[:44]}")


if __name__ == "__main__":
    main()
