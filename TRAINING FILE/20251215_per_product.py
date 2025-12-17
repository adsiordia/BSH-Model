# 20251215_per_product.py
import os, re, json, time, argparse, warnings, hashlib
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, accuracy_score,
    matthews_corrcoef, brier_score_loss, log_loss
)
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

# XGBoost
from xgboost import XGBClassifier

# RDKit (strongly recommended)
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs as DS
    HAVE_RDKIT = True
except Exception:
    HAVE_RDKIT = False

import h5py


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Per-product (one-row-per-ProductName) enzyme holdout binary training")

    # Keep SAME default paths as before
    ap.add_argument("--data_dir", type=str, default="/home/adsiordia/BSH-Model")
    ap.add_argument("--out_dir",  type=str, default="/home/adsiordia/BSH-Model/outputs")
    ap.add_argument("--run_tag",  type=str, default="enzyme_holdout_binary_per_product")

    ap.add_argument("--heat_csv", type=str, default="ipsita_heatmap_long.csv")
    ap.add_argument("--h5_path",  type=str, default="Seqs_list_total.h5")
    ap.add_argument("--enum_xlsx", type=str, default="swap_enumeration_with_core_smiles.xlsx")

    ap.add_argument("--intensity_threshold", type=float, default=1000.0)

    ap.add_argument("--holdout_repeats", type=int, default=5)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac",   type=float, default=0.15)
    ap.add_argument("--test_frac",  type=float, default=0.15)

    ap.add_argument("--seed", type=int, default=42)

    # fingerprints
    ap.add_argument("--amine_fp_bits", type=int, default=1024)
    ap.add_argument("--fp_radius",     type=int, default=2)

    # models
    ap.add_argument("--rf_trees", type=int, default=500)
    ap.add_argument("--mlp_max_iter", type=int, default=200)
    ap.add_argument("--xgb_estimators", type=int, default=1200)

    return ap.parse_args()


# -----------------------------
# Utilities
# -----------------------------
def pjoin(*a): return os.path.join(*a)

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)

def norm_key(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()

def last_token(code: str) -> str:
    return str(code).split("_")[-1]

def split_products(x) -> List[str]:
    if pd.isna(x): return []
    return [p.strip() for p in str(x).split(";") if p.strip()]

def stable_hash_int(s: str) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16)

def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    classes = np.unique(y)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(wi) for c, wi in zip(classes, w)}

def morgan_fp(smiles: str, nBits=1024, radius=2) -> np.ndarray:
    out = np.zeros((nBits,), dtype=np.float32)
    if (not HAVE_RDKIT) or smiles is None:
        return out
    s = str(smiles).strip()
    if not s or s.lower() == "nan":
        return out
    m = Chem.MolFromSmiles(s)
    if m is None:
        return out
    bv = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.int8)
    DS.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)

def metrics_binary(y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_hat = (y_score >= thr).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_hat)),
        "mcc": float(matthews_corrcoef(y_true, y_hat)),
        "acc": float(accuracy_score(y_true, y_hat)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "logloss": float(log_loss(y_true, np.clip(y_score, 1e-7, 1-1e-7))),
        "pos_rate": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }


# -----------------------------
# H5 enzyme embeddings
# -----------------------------
def collect_h5_datasets(h5_path: str) -> List[str]:
    paths = []
    with h5py.File(h5_path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                paths.append(name)
        f.visititems(visit)
    return paths

def match_dataset_for_enzyme(enzyme_id: str, normbase_to_path: Dict[str,str], all_norm_bases: List[str]) -> Optional[str]:
    full_norm = norm_key(enzyme_id)
    tok_norm  = norm_key(last_token(enzyme_id))

    if full_norm in normbase_to_path:
        return normbase_to_path[full_norm]
    if tok_norm in normbase_to_path:
        return normbase_to_path[tok_norm]

    hits = [b for b in all_norm_bases if tok_norm in b or b in tok_norm]
    if hits:
        return normbase_to_path[sorted(hits, key=len)[0]]

    from difflib import get_close_matches
    cand = get_close_matches(tok_norm, all_norm_bases, n=1, cutoff=0.92)
    return normbase_to_path[cand[0]] if cand else None

def load_enzyme_embeddings(h5_path: str, enzyme_ids: List[str]) -> Dict[str, np.ndarray]:
    all_paths = collect_h5_datasets(h5_path)
    if not all_paths:
        raise RuntimeError(f"No datasets found in H5: {h5_path}")

    bases = [p.rsplit("/", 1)[-1] for p in all_paths]
    norm_bases = [norm_key(b) for b in bases]
    normbase_to_path = {}
    for base, nb, full in zip(bases, norm_bases, all_paths):
        normbase_to_path.setdefault(nb, full)
    all_norm_bases = list(normbase_to_path.keys())

    out = {}
    with h5py.File(h5_path, "r") as f:
        for eid in enzyme_ids:
            path = match_dataset_for_enzyme(eid, normbase_to_path, all_norm_bases)
            if path is None:
                continue
            arr = np.array(f[path])
            if arr.ndim == 2 and arr.shape[0] > 1:
                vec = arr.mean(axis=0).astype(np.float32)
            elif arr.ndim == 2 and arr.shape[0] == 1:
                vec = arr[0].astype(np.float32)
            elif arr.ndim > 2:
                vec = arr.reshape(arr.shape[-1]).astype(np.float32)
            else:
                vec = arr.astype(np.float32)
            out[str(eid)] = vec
    if not out:
        raise RuntimeError("No enzyme embeddings loaded. Check enzyme IDs and H5 structure.")
    return out


# -----------------------------
# BA parsing: class + positional tokens
# -----------------------------
def extract_hydroxyl_tokens_from_coretype(core_type: str) -> Tuple[str, ...]:
    """
    Core-Type example: '3a,7a,12a' or '3a, 7b'
    Keep hydroxyl tokens (digits + a/b), drop ketones containing 'k' (e.g., 12k).
    """
    if core_type is None:
        return tuple()
    s0 = str(core_type).strip()
    if not s0 or s0.lower() == "nan":
        return tuple()
    s = s0.lower().replace(" ", "")
    parts = [p for p in s.split(",") if p]
    toks = [p for p in parts if "k" not in p]         # ignore ketones
    toks = [re.sub(r"[^0-9ab]", "", t) for t in toks] # keep digits + a/b only
    toks = [t for t in toks if t]
    return tuple(sorted(set(toks)))

def hydroxyl_class(tokens: Tuple[str, ...]) -> str:
    n = len(tokens)
    if n <= 1: return "mono"
    if n == 2: return "di"
    return "tri"


# -----------------------------
# Build product feature table INSIDE script
# -----------------------------
def build_product_features(enum_xlsx: str,
                           heat_products: set,
                           amine_fp_bits: int,
                           fp_radius: int):
    """
    Returns:
      product_table: one-row-per-ProductName table with BA class one-hot + BA positional one-hot (unambiguous only)
      amine_fps: Morgan FP per product_table row (aligned to product_table order)
      pos_vocab, cls_list
    """
    swap = pd.read_excel(enum_xlsx).copy()

    for c in ["ProductName", "Amine_SMILES", "Core-Type"]:
        if c not in swap.columns:
            swap[c] = np.nan

    # explode ProductName
    swap["ProductName_list"] = swap["ProductName"].apply(split_products)
    ex = swap.explode("ProductName_list").rename(columns={"ProductName_list": "ProductName_clean"})
    ex["ProductName_clean"] = ex["ProductName_clean"].astype(str).str.strip()
    ex = ex[ex["ProductName_clean"].ne("")].copy()

    # restrict to products actually in heatmap
    ex = ex[ex["ProductName_clean"].isin(heat_products)].copy()

    # parse tokens
    ex["hydroxyl_tokens"] = ex["Core-Type"].apply(extract_hydroxyl_tokens_from_coretype)
    ex["token_str"] = ex["hydroxyl_tokens"].apply(lambda t: "|".join(t) if t else "")

    def uniq_list(vals):
        s = set([v for v in vals if v is not None])
        return sorted(s)

    # ambiguity per product
    ba_map = (
        ex.groupby("ProductName_clean", as_index=False)
          .agg(token_str_set=("token_str", uniq_list))
    )
    ba_map["ba_coretype_nunique"] = ba_map["token_str_set"].apply(len)
    ba_map["ba_coretype_ambiguous"] = ba_map["ba_coretype_nunique"] > 1

    # class ambiguity per product
    def class_set(token_str_set):
        classes = set()
        for s in token_str_set:
            toks = tuple([t for t in str(s).split("|") if t])
            classes.add(hydroxyl_class(toks))
        return sorted(classes)

    ba_map["class_set"] = ba_map["token_str_set"].apply(class_set)
    ba_map["class_nunique"] = ba_map["class_set"].apply(len)
    ba_map["class_ambiguous"] = ba_map["class_nunique"] > 1
    ba_map["BA_class"] = ba_map.apply(lambda r: r["class_set"][0] if len(r["class_set"]) == 1 else "unknown", axis=1)

    # positional tokens only if unambiguous
    ba_map["BA_pos_tokens"] = ba_map.apply(
        lambda r: r["token_str_set"][0] if len(r["token_str_set"]) == 1 else "",
        axis=1
    )

    # vocab from unambiguous only
    unamb = ba_map[~ba_map["ba_coretype_ambiguous"]]
    pos_vocab = sorted({tok for s in unamb["BA_pos_tokens"].tolist() for tok in [t for t in str(s).split("|") if t]})

    # BA class one-hot
    cls_list = ["mono", "di", "tri", "unknown"]
    for c in cls_list:
        ba_map[f"ba_class_{c}"] = (ba_map["BA_class"] == c).astype(float)

    # BA positional one-hot
    for tok in pos_vocab:
        ba_map[f"ba_pos_{tok}"] = ba_map["BA_pos_tokens"].apply(lambda s: float(tok in str(s).split("|")))

    # Canonical amine SMILES per product (you verified no multi-mapping)
    canon_amine = (
        ex.groupby("ProductName_clean", as_index=False)
          .agg(Amine_SMILES=("Amine_SMILES", lambda s: next((v for v in s if pd.notna(v) and str(v).strip()), np.nan)))
    )

    product_table = canon_amine.merge(
        ba_map[["ProductName_clean","ba_coretype_ambiguous","class_ambiguous","BA_class","BA_pos_tokens"]
               + [f"ba_class_{c}" for c in cls_list]
               + [f"ba_pos_{t}" for t in pos_vocab]],
        on="ProductName_clean",
        how="left"
    ).rename(columns={"ProductName_clean":"ProductName"})

    # Compute amine FP aligned to product_table order
    amine_fps = np.zeros((len(product_table), amine_fp_bits), dtype=np.float32)
    for i, smi in enumerate(product_table["Amine_SMILES"].tolist()):
        amine_fps[i] = morgan_fp(smi, nBits=amine_fp_bits, radius=fp_radius)

    return product_table, amine_fps, pos_vocab, cls_list


# -----------------------------
# Dataset build: collapse replicates, join product features
# -----------------------------
def derive_label_from_intensity(df: pd.DataFrame, thr: float) -> np.ndarray:
    inten = pd.to_numeric(df["Intensity"], errors="coerce").fillna(-1.0)
    return (inten > float(thr)).astype(int).values

def build_training_matrix(heat_long: pd.DataFrame,
                          enzyme_vecs: Dict[str,np.ndarray],
                          product_table: pd.DataFrame,
                          amine_fps: np.ndarray,
                          pos_vocab: List[str],
                          cls_list: List[str],
                          thr: float):
    hl = heat_long.copy()
    hl["Enzyme"] = hl["Enzyme"].astype(str).str.strip()
    hl["ProductName"] = hl["ProductName"].astype(str).str.strip()

    # Keep only rows with enzyme embedding
    hl = hl[hl["Enzyme"].isin(enzyme_vecs.keys())].copy()

    # Label and collapse replicates
    hl["y"] = derive_label_from_intensity(hl, thr=thr)
    hl = hl.groupby(["Enzyme","ProductName"], as_index=False).agg(y=("y","max"))

    # Join product features
    prod = product_table.copy()
    prod["ProductName"] = prod["ProductName"].astype(str).str.strip()
    df = hl.merge(prod, on="ProductName", how="left")

    # Drop missing amine smiles (failed join or missing)
    df = df[df["Amine_SMILES"].notna()].copy()

    # Build blocks
    dE = next(iter(enzyme_vecs.values())).shape[0]
    E = np.zeros((len(df), dE), dtype=np.float32)
    for i, e in enumerate(df["Enzyme"].tolist()):
        E[i] = enzyme_vecs[e]

    # Align amine_fps by ProductName index in product_table
    prod_index = {pn:i for i, pn in enumerate(prod["ProductName"].tolist())}
    A = np.zeros((len(df), amine_fps.shape[1]), dtype=np.float32)
    for i, pn in enumerate(df["ProductName"].tolist()):
        A[i] = amine_fps[prod_index[pn]]

    # BA class one-hot
    C = df[[f"ba_class_{c}" for c in cls_list]].to_numpy(dtype=np.float32)

    # BA positional one-hot (unambiguous only; ambiguous rows have zeros)
    P = df[[f"ba_pos_{t}" for t in pos_vocab]].to_numpy(dtype=np.float32) if len(pos_vocab) else np.zeros((len(df),0), dtype=np.float32)

    # Add ambiguity flag as feature
    amb_flag = df[["ba_coretype_ambiguous"]].fillna(False).astype(float).to_numpy(dtype=np.float32)

    X = np.concatenate([E, A, C, P, amb_flag], axis=1)
    y = df["y"].astype(int).to_numpy()
    groups = df["Enzyme"].astype(str).to_numpy(dtype=object)

    meta = {
        "dims": {
            "enzyme_emb": int(E.shape[1]),
            "amine_fp": int(A.shape[1]),
            "ba_class": int(C.shape[1]),
            "ba_pos": int(P.shape[1]),
            "amb_flag": int(amb_flag.shape[1]),
            "X_total": int(X.shape[1]),
        },
        "pos_vocab": pos_vocab,
        "class_order": cls_list,
        "rdkit_available": HAVE_RDKIT,
        "threshold": float(thr),
    }
    return X, y, groups, df, meta


# -----------------------------
# Holdout splitting by enzyme (no leakage)
# -----------------------------
def enzyme_holdout_split(groups: np.ndarray, train_frac: float, val_frac: float, test_frac: float, seed: int):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)

    n = len(uniq)
    n_train = int(round(train_frac * n))
    n_val   = int(round(val_frac * n))

    enz_train = set(uniq[:n_train])
    enz_val   = set(uniq[n_train:n_train+n_val])
    enz_test  = set(uniq[n_train+n_val:])

    train_mask = np.array([g in enz_train for g in groups], dtype=bool)
    val_mask   = np.array([g in enz_val   for g in groups], dtype=bool)
    test_mask  = np.array([g in enz_test  for g in groups], dtype=bool)

    return train_mask, val_mask, test_mask, (enz_train, enz_val, enz_test)


# -----------------------------
# Fit models + learning curves
# -----------------------------
def fit_xgb_with_curve(Xtr, ytr, Xva, yva, seed: int, n_estimators: int):
    neg = float((ytr == 0).sum()); pos = float((ytr == 1).sum())
    spw = max(neg / max(pos, 1.0), 1.0)

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=7,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        objective="binary:logistic",
        tree_method="hist",
        max_bin=256,
        eval_metric=["aucpr","logloss"],
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=spw,
    )

    t0 = time.time()
    clf.fit(
        Xtr, ytr,
        eval_set=[(Xtr, ytr), (Xva, yva)],
        verbose=False
    )
    dur = time.time() - t0

    hist = clf.evals_result()
    curve = pd.DataFrame({
        "round": np.arange(len(hist["validation_0"]["logloss"])),
        "train_logloss": hist["validation_0"]["logloss"],
        "val_logloss": hist["validation_1"]["logloss"],
        "train_aucpr": hist["validation_0"]["aucpr"],
        "val_aucpr": hist["validation_1"]["aucpr"],
    })
    return clf, dur, curve

def fit_rf(Xtr, ytr, seed: int, n_trees: int):
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=n_trees,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(Xtr, ytr)
    dur = time.time() - t0
    return clf, dur

def fit_mlp_with_curve(Xtr, ytr, Xva, yva, seed: int, max_iter: int):
    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr).astype(np.float32)
    Xva_s = scaler.transform(Xva).astype(np.float32)

    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=max_iter,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=seed,
    )

    t0 = time.time()
    clf.fit(Xtr_s, ytr)
    dur = time.time() - t0

    lc = getattr(clf, "loss_curve_", None)
    curve = None
    if lc is not None:
        curve = pd.DataFrame({"epoch": np.arange(len(lc)), "train_loss": lc})

    return clf, scaler, dur, curve


# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    out_root = pjoin(os.path.abspath(args.out_dir), args.run_tag)
    os.makedirs(out_root, exist_ok=True)

    heat_path = pjoin(args.data_dir, args.heat_csv) if not os.path.isabs(args.heat_csv) else args.heat_csv
    h5_path   = pjoin(args.data_dir, args.h5_path) if not os.path.isabs(args.h5_path) else args.h5_path
    enum_path = pjoin(args.data_dir, args.enum_xlsx) if not os.path.isabs(args.enum_xlsx) else args.enum_xlsx

    # Load heatmap
    heat = pd.read_csv(heat_path)
    required = {"Enzyme","ProductName","Intensity"}
    if not required.issubset(set(heat.columns)):
        raise ValueError(f"Heat CSV must include columns: {sorted(required)}")

    heat["Enzyme"] = heat["Enzyme"].astype(str).str.strip()
    heat["ProductName"] = heat["ProductName"].astype(str).str.strip()

    heat_products = set(heat["ProductName"].unique().tolist())
    enzyme_ids = sorted(heat["Enzyme"].unique().tolist())

    print("== Config ==")
    print("heat:", heat_path)
    print("h5:", h5_path)
    print("enum:", enum_path)
    print("threshold:", args.intensity_threshold)
    print("RDKit:", HAVE_RDKIT)
    if not HAVE_RDKIT:
        print("WARNING: RDKit not available -> amine fingerprints will be all-zeros.")

    # Load enzyme embeddings
    enzyme_vecs = load_enzyme_embeddings(h5_path, enzyme_ids)
    print(f"[Embeddings] loaded={len(enzyme_vecs)} dim={next(iter(enzyme_vecs.values())).shape[0]}")

    # Build product features (one row per ProductName)
    product_table, amine_fps, pos_vocab, cls_list = build_product_features(
        enum_xlsx=enum_path,
        heat_products=heat_products,
        amine_fp_bits=args.amine_fp_bits,
        fp_radius=args.fp_radius
    )
    product_table.to_csv(pjoin(out_root, "product_features_one_row_per_product.csv"), index=False)
    print(f"[Product table] rows={len(product_table)} unique products={product_table['ProductName'].nunique()}")
    print(f"[BA pos vocab] {pos_vocab}")

    # Build training matrix
    X, y, groups, aligned_df, feature_meta = build_training_matrix(
        heat_long=heat,
        enzyme_vecs=enzyme_vecs,
        product_table=product_table,
        amine_fps=amine_fps,
        pos_vocab=pos_vocab,
        cls_list=cls_list,
        thr=args.intensity_threshold
    )

    # Save meta
    with open(pjoin(out_root, "feature_meta.json"), "w") as f:
        json.dump(feature_meta, f, indent=2)

    pos = int((y==1).sum()); neg = int((y==0).sum())
    print(f"[Dataset] X={X.shape} enzymes={len(np.unique(groups))} pos={pos} ({pos/len(y):.3f}) neg={neg}")

    # Holdout repeats
    metrics_rows = []
    preds_rows = []

    run_meta = {
        "run_tag": args.run_tag,
        "threshold": args.intensity_threshold,
        "holdout_repeats": args.holdout_repeats,
        "fractions": {"train": args.train_frac, "val": args.val_frac, "test": args.test_frac},
        "seed": args.seed,
        "paths": {"heat": heat_path, "h5": h5_path, "enum": enum_path},
        "rdkit_available": HAVE_RDKIT,
        "models": {
            "xgb": {"n_estimators": args.xgb_estimators},
            "rf":  {"n_trees": args.rf_trees},
            "mlp": {"max_iter": args.mlp_max_iter},
        },
    }

    for rep in range(1, args.holdout_repeats + 1):
        print(f"\n== Holdout repeat {rep}/{args.holdout_repeats} ==")
        train_mask, val_mask, test_mask, (enz_tr, enz_val, enz_te) = enzyme_holdout_split(
            groups, args.train_frac, args.val_frac, args.test_frac, seed=args.seed + rep
        )

        Xtr, ytr = X[train_mask], y[train_mask]
        Xva, yva = X[val_mask], y[val_mask]
        Xte, yte = X[test_mask], y[test_mask]

        print(f"  rows train/val/test = {len(ytr)}/{len(yva)}/{len(yte)}")
        print(f"  pos% train/val/test = {np.mean(ytr):.3f}/{np.mean(yva):.3f}/{np.mean(yte):.3f}")
        print(f"  enzymes train/val/test = {len(enz_tr)}/{len(enz_val)}/{len(enz_te)}")

        # ---- XGB ----
        xgb_model, xgb_time, xgb_curve = fit_xgb_with_curve(
            Xtr, ytr, Xva, yva, seed=args.seed+rep, n_estimators=args.xgb_estimators
        )
        xgb_curve.to_csv(pjoin(out_root, f"holdout_rep{rep}_xgb_learning_curve.csv"), index=False)

        p_val = xgb_model.predict_proba(Xva)[:,1]
        p_te  = xgb_model.predict_proba(Xte)[:,1]
        m_val = metrics_binary(yva, p_val)
        m_te  = metrics_binary(yte, p_te)

        metrics_rows.append({
            "rep": rep, "model": "xgb", "train_time_sec": xgb_time,
            **{f"val_{k}": v for k,v in m_val.items()},
            **{f"test_{k}": v for k,v in m_te.items()},
        })
        preds_rows.append(pd.DataFrame({"rep": rep, "model": "xgb", "y_true": yte, "y_score": p_te}))

        # ---- RF ----
        rf_model, rf_time = fit_rf(Xtr, ytr, seed=args.seed+rep, n_trees=args.rf_trees)
        p_val = rf_model.predict_proba(Xva)[:,1]
        p_te  = rf_model.predict_proba(Xte)[:,1]
        m_val = metrics_binary(yva, p_val)
        m_te  = metrics_binary(yte, p_te)

        metrics_rows.append({
            "rep": rep, "model": "rf", "train_time_sec": rf_time,
            **{f"val_{k}": v for k,v in m_val.items()},
            **{f"test_{k}": v for k,v in m_te.items()},
        })
        preds_rows.append(pd.DataFrame({"rep": rep, "model": "rf", "y_true": yte, "y_score": p_te}))

        # ---- MLP ----
        mlp_model, mlp_scaler, mlp_time, mlp_curve = fit_mlp_with_curve(
            Xtr, ytr, Xva, yva, seed=args.seed+rep, max_iter=args.mlp_max_iter
        )
        if mlp_curve is not None:
            mlp_curve.to_csv(pjoin(out_root, f"holdout_rep{rep}_mlp_learning_curve.csv"), index=False)

        Xva_s = mlp_scaler.transform(Xva).astype(np.float32)
        Xte_s = mlp_scaler.transform(Xte).astype(np.float32)
        p_val = mlp_model.predict_proba(Xva_s)[:,1]
        p_te  = mlp_model.predict_proba(Xte_s)[:,1]
        m_val = metrics_binary(yva, p_val)
        m_te  = metrics_binary(yte, p_te)

        metrics_rows.append({
            "rep": rep, "model": "mlp", "train_time_sec": mlp_time,
            **{f"val_{k}": v for k,v in m_val.items()},
            **{f"test_{k}": v for k,v in m_te.items()},
        })
        preds_rows.append(pd.DataFrame({"rep": rep, "model": "mlp", "y_true": yte, "y_score": p_te}))

    metrics_df = pd.DataFrame(metrics_rows)
    preds_df = pd.concat(preds_rows, ignore_index=True)

    metrics_df.to_csv(pjoin(out_root, "metrics_per_repeat.csv"), index=False)
    preds_df.to_csv(pjoin(out_root, "predictions_test.csv"), index=False)

    # summary (mean/std per model)
    numeric_cols = [c for c in metrics_df.columns if c not in ("rep","model")]
    summary = (
        metrics_df.groupby("model", as_index=False)[numeric_cols]
        .agg(["mean","std"])
    )
    summary.columns = ["_".join([x for x in col if x]) for col in summary.columns.values]
    summary.to_csv(pjoin(out_root, "metrics_summary.csv"), index=False)

    with open(pjoin(out_root, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    print("\n=== Saved ===")
    print("[Saved]", pjoin(out_root, "product_features_one_row_per_product.csv"))
    print("[Saved]", pjoin(out_root, "feature_meta.json"))
    print("[Saved]", pjoin(out_root, "metrics_per_repeat.csv"))
    print("[Saved]", pjoin(out_root, "metrics_summary.csv"))
    print("[Saved]", pjoin(out_root, "predictions_test.csv"))
    print("[Saved]", pjoin(out_root, "run_meta.json"))
    print("\nDone. Root outputs:", out_root)


if __name__ == "__main__":
    main()
