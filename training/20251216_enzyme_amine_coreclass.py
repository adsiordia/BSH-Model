#!/usr/bin/env python

import os, re, json, time, argparse, warnings, hashlib
import numpy as np
import pandas as pd
import h5py

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    accuracy_score, matthews_corrcoef, brier_score_loss, log_loss
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================
# Utilities
# ============================

def norm_key(s):
    return re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()

def last_token(s):
    return str(s).split("_")[-1]

def set_seed(seed):
    np.random.seed(seed)

# ============================
# RDKit (amine FP)
# ============================

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs
    HAVE_RDKIT = True
except:
    HAVE_RDKIT = False

def morgan_fp(smiles, nBits=1024, radius=2):
    out = np.zeros(nBits, dtype=np.float32)
    if not HAVE_RDKIT or not smiles or pd.isna(smiles):
        return out
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return out
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    DataStructs.ConvertToNumpyArray(bv, out)
    return out

# ============================
# Enzyme embeddings (ROBUST)
# ============================

def collect_h5_datasets(h5_path):
    paths = []
    with h5py.File(h5_path, "r") as f:
        f.visititems(lambda n, o: paths.append(n) if isinstance(o, h5py.Dataset) else None)
    return paths

def load_enzyme_embeddings(h5_path, enzyme_ids):
    all_paths = collect_h5_datasets(h5_path)
    bases = [p.rsplit("/",1)[-1] for p in all_paths]
    norm_bases = [norm_key(b) for b in bases]

    norm2path = {}
    for b, nb, p in zip(bases, norm_bases, all_paths):
        norm2path.setdefault(nb, p)

    def match(eid):
        fn = norm_key(eid)
        tk = norm_key(last_token(eid))
        if fn in norm2path: return norm2path[fn]
        if tk in norm2path: return norm2path[tk]
        hits = [b for b in norm_bases if tk in b or b in tk]
        if hits:
            return norm2path[min(hits, key=len)]
        return None

    out = {}
    with h5py.File(h5_path, "r") as f:
        for eid in enzyme_ids:
            p = match(eid)
            if p is None: continue
            arr = np.array(f[p])
            if arr.ndim == 2:
                arr = arr.mean(axis=0)
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[-1])
            out[eid] = arr.astype(np.float32)

    print(f"[Embeddings] matched {len(out)} / {len(enzyme_ids)} enzymes")
    return out

# ============================
# BA core class (mono/di/tri)
# ============================

CLS = ["mono", "di", "tri"]

def extract_core_class(core_type):
    if pd.isna(core_type): return "mono"
    toks = [t for t in str(core_type).lower().replace(" ","").split(",") if "k" not in t]
    if len(toks) <= 1: return "mono"
    if len(toks) == 2: return "di"
    return "tri"

def core_onehot(core_type):
    cls = extract_core_class(core_type)
    v = np.zeros(3, dtype=np.float32)
    v[CLS.index(cls)] = 1.0
    return v

# ============================
# Metrics
# ============================

def metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    return {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
        "f1": f1_score(y, yhat),
        "mcc": matthews_corrcoef(y, yhat),
        "acc": accuracy_score(y, yhat),
        "brier": brier_score_loss(y, p),
        "logloss": log_loss(y, p),
    }

# ============================
# MAIN
# ============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/adsiordia/BSH-Model")
    ap.add_argument("--out_dir", default="/home/adsiordia/BSH-Model/outputs/enzyme_amine_coreclass")
    ap.add_argument("--heat_csv", default="ipsita_heatmap_long.csv")
    ap.add_argument("--h5_path", default="Seqs_list_total.h5")
    ap.add_argument("--swap_xlsx", default="swap_enumeration_with_core_smiles.xlsx")
    ap.add_argument("--intensity_threshold", type=float, default=1000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # ---- Load data
    heat = pd.read_csv(os.path.join(args.data_dir, args.heat_csv))
    swap = pd.read_excel(os.path.join(args.data_dir, args.swap_xlsx))

    heat["label"] = (heat["Intensity"] > args.intensity_threshold).astype(int)

    enzymes = heat["Enzyme"].astype(str).unique().tolist()
    emb = load_enzyme_embeddings(os.path.join(args.data_dir, args.h5_path), enzymes)

    swap = swap.drop_duplicates("ProductName")
    swap["BA_class"] = swap["Core-Type"].apply(extract_core_class)

    df = heat.merge(swap, on="ProductName", how="inner")
    df = df[df["Enzyme"].isin(emb)]

    # ---- Features
    E = np.vstack([emb[e] for e in df["Enzyme"]])
    A = np.vstack([morgan_fp(s) for s in df["Amine_SMILES"]])
    B = np.vstack([core_onehot(c) for c in df["Core-Type"]])

    X = np.hstack([E, A, B])
    y = df["label"].values
    groups = df["Enzyme"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # ---- Models
    MODELS = {
        "xgb": XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.7,
            objective="binary:logistic",
            eval_metric=["logloss","aucpr"],
            tree_method="hist", random_state=args.seed
        ),
        "rf": RandomForestClassifier(n_estimators=400, class_weight="balanced"),
        "mlp": MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300,
                             early_stopping=True, random_state=args.seed)
    }

    rows = []
    preds_all = []

    splitter = GroupShuffleSplit(n_splits=args.repeats, test_size=0.2, random_state=args.seed)

    for rep, (tr, te) in enumerate(splitter.split(Xs, y, groups), 1):
        for name, model in MODELS.items():
            Xtr, Xte = Xs[tr], Xs[te]
            ytr, yte = y[tr], y[te]

            t0 = time.time()

            if name == "xgb":
                model.fit(Xtr, ytr, eval_set=[(Xtr,ytr),(Xte,yte)], verbose=False)
                hist = model.evals_result()
                keys = list(hist.keys())
                pd.DataFrame({
                    "round": range(len(hist[keys[0]]["logloss"])),
                    "train_logloss": hist[keys[0]]["logloss"],
                    "val_logloss": hist[keys[1]]["logloss"] if len(keys)>1 else None
                }).to_csv(f"{args.out_dir}/xgb_learning_curve_rep{rep}.csv", index=False)
            else:
                model.fit(Xtr, ytr)

            p = model.predict_proba(Xte)[:,1]
            m = metrics(yte, p)

            rows.append({
                "rep": rep, "model": name, **m,
                "time": time.time()-t0
            })

            preds_all.append(pd.DataFrame({
                "rep": rep, "model": name,
                "y_true": yte, "y_score": p
            }))

    pd.DataFrame(rows).to_csv(f"{args.out_dir}/metrics.csv", index=False)
    pd.concat(preds_all).to_csv(f"{args.out_dir}/predictions.csv", index=False)

    print("=== DONE ===")
    print(f"Saved to {args.out_dir}")

if __name__ == "__main__":
    main()
