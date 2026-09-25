"""Protein vectors for Horizyn, two variants, matched by UniProt accession.

  full    mean over every residue of the whole sequence  -- all 115 + extras
  mature  mean over the residues AFTER the signal peptide -- the 65 trimmed

`mature` is not identical to embedding the trimmed sequence: ProtT5 is a
transformer, so the signal peptide still shaped the representation of the
residues that follow it. It isolates the mature region without re-running the
model, which is the closest we can get from what is on disk.
"""
from pathlib import Path
import sys, h5py, json, numpy as np
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT/"src"))
import labels as lb
SRC = Path("/home/adsiordia/BSH-Model-v2/data/raw/Seqs_list_total_per_residue.h5")

cut = {r["accession"]: r["cut"] for r in
       json.loads((ROOT/"site/candidates.json").read_text())["proteins"]
       if r.get("cut") is not None}
panel = set(lb.build(min_reps=1).Enzyme)

full, mature, skipped = {}, {}, []
with h5py.File(SRC, "r") as f:
    for k in f.keys():
        acc = k.split("_")[-1]
        a = np.asarray(f[k], dtype=np.float32)
        full[acc] = a.mean(0)
        c = cut.get(acc)
        if c is None:
            continue
        if c >= a.shape[0] - 20:
            skipped.append((acc, f"cut {c} of {a.shape[0]} residues")); continue
        mature[acc] = a[c:].mean(0)

def write(name, d):
    ids = sorted(d)
    V = np.vstack([d[a] for a in ids]).astype(np.float32)
    p = ROOT/f"outputs/horizyn_proteins_{name}.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("ids", data=np.array(ids, dtype=h5py.string_dtype()))
        f.create_dataset("vectors", data=V)
    print(f"{name:7s}: {len(ids):3d} proteins, {V.shape}  -> {p.name}")
    return ids, V

fi, Vf = write("full", full)
mi, Vm = write("mature", mature)
print(f"\nfull covers the 115-enzyme panel: {len(set(fi) & panel)} of {len(panel)}")
if skipped: print(f"skipped for mature: {skipped}")

both = sorted(set(fi) & set(mi))
xf = {a: i for i, a in enumerate(fi)}; xm = {a: i for i, a in enumerate(mi)}
cs = np.array([np.dot(Vf[xf[a]], Vm[xm[a]]) /
               (np.linalg.norm(Vf[xf[a]]) * np.linalg.norm(Vm[xm[a]])) for a in both])
print(f"\ncosine(full, mature) for the {len(both)} with both: "
      f"median {np.median(cs):.4f}, min {cs.min():.4f}")
