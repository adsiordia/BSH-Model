"""Score our BSH enzymes against our BSH reactions with the Horizyn model.

Horizyn is a dual encoder: both sides are projected to 512 dimensions and
L2-normalised, so a score is a COSINE SIMILARITY in [-1, 1], not a probability.
It is only meaningful as a ranking -- here, which of the 75 amine x core
conjugations best suits each enzyme.

    horizyn-env/bin/python src/horizyn_score.py
"""
from pathlib import Path
import sys, json, numpy as np, pandas as pd, torch, h5py
from tqdm import tqdm

REPO = Path("/home/adsiordia/horizyn-repo")
sys.path.insert(0, str(REPO))
from horizyn.config import load_config
from horizyn.datasets.base import BaseDataset
from horizyn.datasets.collection import MergeDataset
from horizyn.datasets.fingerprints import DRFPFingerprintDataset, RDKitPlusFingerprintDataset
from horizyn.datasets.transform import ConcatTensorTransform
from horizyn.lightning_module import HorizynLitModule

ROOT = Path(__file__).resolve().parent.parent
CKPT = REPO / "checkpoints/horizyn_v1_0_inf.ckpt"
DEV = "cpu"


def fingerprints(smiles_list, config):
    """RDKit+ (1024) and DRFP (1024) concatenated, exactly as scripts/predict.py."""
    keys = [f"r{i}" for i in range(len(smiles_list))]
    ds = BaseDataset(keys=keys, array_data=[{"reaction_smiles": s} for s in smiles_list])
    common = dict(standardize=config.data.get("standardize_reactions", True),
                  standardize_hypervalent=config.data.get("standardize_hypervalent", True),
                  standardize_remove_hs=config.data.get("standardize_remove_hs", True),
                  standardize_kekulize=config.data.get("standardize_kekulize", False),
                  standardize_uncharge=config.data.get("standardize_uncharge", True),
                  standardize_metals=config.data.get("standardize_metals", True))
    rd = RDKitPlusFingerprintDataset(reaction_dataset=ds,
        vec_dim=config.data.get("rdkit_fp_dim", 1024), mol_fp_type="morgan",
        rxn_fp_type="struct", use_chirality=True, **common)
    dr = DRFPFingerprintDataset(reaction_dataset=ds,
        vec_dim=config.data.get("drfp_dim", 1024), radius=3, rings=True, **common)
    m = MergeDataset(datasets={"rdkit": rd, "drfp": dr}, add_prefix=False)
    m.append_transforms(ConcatTensorTransform(labels=["rdkit", "drfp"], dim=0))
    return torch.stack([m[k] for k in tqdm(keys, desc="fingerprints")])


def main():
    rx = pd.read_csv(ROOT / "outputs/horizyn_reactions.csv")
    print(f"{len(rx)} reactions, {rx.amine.nunique()} amines x {rx.core.nunique()} cores "
          f"x {rx.leaving_group.nunique()} leaving groups")

    print(f"\nloading {CKPT.name}")
    model = HorizynLitModule.load_from_checkpoint(str(CKPT), map_location=DEV)
    model.eval()
    # the same config scripts/predict.py loads -- it carries the fingerprint
    # settings the model was trained with, so they must match exactly
    config = load_config(str(REPO / "configs/sota.yaml"))
    print(f"  query encoder  in {model.model.query_encoder.input_dim} "
          f"-> out {model.model.query_encoder.output_dim}")
    print(f"  target encoder in {model.model.target_encoder.input_dim} "
          f"-> out {model.model.target_encoder.output_dim}")

    fps = fingerprints(rx.reaction.tolist(), config)
    print(f"reaction fingerprints {tuple(fps.shape)}")
    with torch.no_grad():
        R = model.model.query_encoder(fps.to(DEV))
    print(f"  reaction vectors unit-norm: "
          f"{bool(torch.allclose(R.norm(dim=1), torch.ones(len(R)), atol=1e-4))}")

    for variant in ("full", "mature"):
        f_in = ROOT / f"outputs/horizyn_proteins_{variant}.h5"
        with h5py.File(f_in, "r") as f:
            pids = [x.decode() if isinstance(x, bytes) else str(x) for x in f["ids"][:]]
            pvec = torch.tensor(np.asarray(f["vectors"][:], dtype=np.float32))
        with torch.no_grad():
            P = model.model.target_encoder(pvec.to(DEV))
            S = (R @ P.T).cpu().numpy()
        print(f"\n[{variant}] {len(pids)} proteins  scores {S.shape}  "
              f"range {S.min():+.4f} to {S.max():+.4f}")
        rows = [dict(accession=acc, amine=r.amine, core=r.core,
                     leaving_group=r.leaving_group, score=float(S[i, j]))
                for j, acc in enumerate(pids) for i, r in rx.iterrows()]
        d = pd.DataFrame(rows)
        avg = (d.groupby(["accession", "amine", "core"], as_index=False)
                 .score.mean().rename(columns={"score": "horizyn"}))
        avg["rank"] = avg.groupby("accession").horizyn.rank(ascending=False).astype(int)
        avg.to_csv(ROOT / f"outputs/horizyn_scores_{variant}.csv", index=False)
        print(f"  wrote outputs/horizyn_scores_{variant}.csv ({len(avg):,} pairs)")


if __name__ == "__main__":
    main()
