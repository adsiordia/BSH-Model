"""One loader for every protein embedding we have, returning whole-protein vectors.

    ProtT5   data/raw/Seqs_list_total_per_residue.h5   (L,1024) per residue
    ESM-3    data/raw/esm3_per_residue.h5              (L,1536) per residue
    ESM-2    ~/esm2_emebddings/bsh_esm2/*.pt           (1280,)  already mean-pooled
    ProstT5  ~/prost5_embeddings/...prost5.h5          (1024,)  already pooled

Two gotchas handled here:
  * ProstT5 is written with h5 dtype int16, but the values are BFLOAT16 bit patterns --
    the writing script sets ds.attrs['dtype'] = 'bfloat16' and casts with
    .to(torch.bfloat16).view(torch.int16). bfloat16 is the top 16 bits of a float32,
    so it is decoded by shifting left 16 and viewing as float32. Reading it as
    float16 instead also yields plausible-looking numbers (-1.8..1.8 rather than the
    true -0.48..0.69), which is why the error survived for a while; the two decodings
    correlate only 0.62.
  * ESM-2 files hold only `mean_representations`, so whole-protein mean is the
    only pooling available for it -- which is why every source is reduced the
    same way here, so a comparison isolates the model rather than the pooling.
"""
from pathlib import Path
import glob, os
import h5py, numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw"
EMB = ROOT / "data/embeddings"
HOME = Path.home()


def _pick(*candidates):
    """First path that exists. Lets a fresh clone use the bundled copy while
    still honouring a local embeddings directory when one is present."""
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return Path(candidates[0])

SOURCES = {
    "ProtT5":  dict(kind="h5_per_residue", path=RAW / "Seqs_list_total_per_residue.h5", dim=1024),
    "ESM-3":   dict(kind="h5_per_residue", path=RAW / "esm3_per_residue.h5",            dim=1536),
    "ESM-2":   dict(kind="pt_mean",        path=HOME / "esm2_emebddings/bsh_esm2",      dim=1280),
    "ProstT5": dict(kind="h5_pooled_bf16",
                    path=_pick(EMB / "Seqs_list_total_prost5.h5",
                               HOME / "prost5_embeddings/Seqs_list_total_prost5.h5"), dim=1024),
}


def _acc(name):
    return name.split("_")[-1]


def _bf16(raw):
    """int16 holding bfloat16 bit patterns -> float32.

    bfloat16 is simply the high half of a float32, so the bits are shifted back up.
    """
    return (raw.astype(np.uint16).astype(np.uint32) << 16).view(np.float32)


def load(source, pooling="mean"):
    """accession -> one vector per protein.

    pooling applies only to per-residue sources; the already-pooled ones ignore it.
    """
    s = SOURCES[source]
    k = s["kind"]
    if k == "h5_per_residue":
        fn = np.mean if pooling == "mean" else np.max
        with h5py.File(s["path"], "r") as f:
            return {_acc(key): fn(f[key][:], axis=0).astype(np.float32) for key in f.keys()}
    if k == "h5_pooled_bf16":
        with h5py.File(s["path"], "r") as f:
            return {_acc(key): _bf16(f[key][:]) for key in f.keys()}
    if k == "pt_mean":
        import torch
        out = {}
        for p in glob.glob(str(Path(s["path"]) / "*.pt")):
            o = torch.load(p, map_location="cpu", weights_only=False)
            v = o["mean_representations"][33] if "mean_representations" in o else o["representations"][33].mean(0)
            out[_acc(os.path.basename(p)[:-3])] = v.numpy().astype(np.float32)
        return out
    raise ValueError(k)


def available(enzymes=None):
    """Which sources cover which enzymes."""
    rows = {}
    for name in SOURCES:
        try:
            e = load(name)
            rows[name] = set(e) if enzymes is None else set(e) & set(enzymes)
        except Exception as exc:
            rows[name] = f"ERROR: {exc}"
    return rows


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import labels as lb
    need = set(lb.build(min_reps=1).Enzyme)
    print(f"{'source':10s} {'dim':>6} {'proteins':>9} {'covers our 115':>15}  value range")
    for name in SOURCES:
        e = load(name)
        v = np.stack(list(e.values()))
        print(f"{name:10s} {v.shape[1]:>6} {len(e):>9} {len(set(e)&need):>15}  "
              f"[{v.min():+.2f}, {v.max():+.2f}] mean {v.mean():+.3f}")
    common = set.intersection(*(set(load(n)) & need for n in SOURCES))
    print(f"\nenzymes covered by ALL four: {len(common)}")
