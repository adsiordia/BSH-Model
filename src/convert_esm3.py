"""Convert the individual ESM-3 .pt files into one HDF5, in the same layout as
the ProtT5 file, so dataset.build(embedding_h5=...) can use either.

Source: /home/adsiordia/esm3/bsh/bsh_esm3_embeddings/*.pt
        each a (sequence_length, 1536) float32 tensor, named <prefix>_<accession>.pt

Only the enzymes present in Trimmed_remove_proteomics.csv are written.

One exclusion: A0A1I4P275's ESM-3 tensor has 300 residues while its FASTA and
ProtT5 embedding have 347. The conservation-column mapping is built from the MSA
of the 347-residue sequence, so applying it to a 300-residue tensor would select
the wrong residues silently. It is dropped rather than risk that, which keeps the
ProtT5/ESM-3 comparison on an identical enzyme set.
"""
import argparse, glob, os
from pathlib import Path

import h5py, numpy as np, torch
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent
SRC = "/home/adsiordia/esm3/bsh/bsh_esm3_embeddings"
OUT = ROOT / "data/raw/esm3_per_residue.h5"


def main(src=SRC, out=OUT, strict=True):
    import labels as lb
    need = set(lb.build(min_reps=1).Enzyme)
    seqs = {r.id.split("_")[-1]: len(str(r.seq).replace("-", ""))
            for r in SeqIO.parse(ROOT / "data/raw/Seqs_list_total.fasta", "fasta")}

    files = {os.path.basename(f)[:-3].split("_")[-1]: f for f in sorted(glob.glob(f"{src}/*.pt"))}
    written, skipped, mismatched = [], [], []
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as h:
        for acc in sorted(need):
            f = files.get(acc)
            if f is None:
                skipped.append(acc); continue
            t = torch.load(f, map_location="cpu", weights_only=False).numpy().astype(np.float32)
            expected = seqs.get(acc)
            if expected is not None and t.shape[0] != expected:
                mismatched.append((acc, t.shape[0], expected))
                if strict:
                    continue
            h.create_dataset(acc, data=t)
            written.append(acc)
    print(f"wrote {out}  ({Path(out).stat().st_size/1e6:.0f} MB)")
    print(f"  {len(written)} enzymes, dim {t.shape[1]}")
    if skipped:
        print(f"  no ESM-3 file for {len(skipped)}: {skipped}")
    for acc, got, exp in mismatched:
        print(f"  {'EXCLUDED' if strict else 'kept'} {acc}: ESM-3 has {got} residues, "
              f"sequence has {exp} -- residue indices would not line up")
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--keep-mismatched", action="store_true")
    a = ap.parse_args()
    main(a.src, Path(a.out), strict=not a.keep_mismatched)
