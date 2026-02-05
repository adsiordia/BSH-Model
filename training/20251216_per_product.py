import os, re, json, time, argparse, warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    log_loss,
    brier_score_loss,
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

# --- XGBoost (use xgboost.train for maximum compatibility) ---
import xgboost as xgb

# --- RDKit optional (recommended). If missing, will fall back to hashed name features for amine ---
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs as DS
    HAVE_RDKIT = True
except Exception:
    HAVE_RDKIT = False

# --- HDF5 enzyme embeddings ---
import h5py


# ============================================================
# Args
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser("Per-product baseline: Enzyme + Amine + mono/di/tri")

    ap.add_argument("--data_dir", type=str, default=os.path.join(os.path.dirname(__file__), "..", "data"))
    ap.add_argument("--out_dir",  type=str, default=os.path.join(os.path.dirname(__file__), "..", "outputs"))
    ap.add_argument("--run-tag", type=str, default="enzyme_amine_class_baseline")

    ap.add_argument("--heat_csv", type=str, default="ipsita_heatmap_long.csv")
    ap.add_argument("--h5_path", type=str, default="Seqs_list_total.h5")
    ap.add_argument("--enum_xlsx", type=str, default="swap_enumeration_with_core_smiles.xlsx")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--intensity-threshold", type=float, default=1000.0)

    # Repeated enzyme holdout
    ap.add_argument("--holdout-repeats", type=int, default=5)
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)

    # Features
    ap.add_argument("--amine-fp-bits", type=int, default=1024)
    ap.add_argument("--fp-radius", type=int, default=2)

    # Models
    ap.add_argument("--no-rf", action="store_true")
    ap.add_argument("--no-mlp", action="store_true")
    ap.add_argument("--no-logreg", action="store_true")
    ap.add_argument("--no-xgb", action="store_true")

    return ap.parse_args()


# ============================================================
# Utils
# ============================================================

def pjoin(*a): return os.path.join(*a)

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)

def norm_key(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()

def last_token(code: str) -> str:
    return str(code).split("_")[-1]

def stable_hash_int(s: str) -> int:
    import hashlib
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16)

def name_hash_fp(name: str, nBits: int = 1024) -> np.ndarray:
    """Simple hashed character 3-gram fingerprint (fallback if RDKit missing or SMILES missing)."""
    s = (str(name) or "").lower().strip()
    t = f"^{s}$"
    grams = set()
    for i in range(max(0, len(t) - 2)):
        grams.add(t[i:i+3])
    bits = np.zeros((nBits,), dtype=np.float32)
    for g in grams:
        idx = stable_hash_int(g) % nBits
        bits[idx] = 1.0
    return bits

def morgan_fp(smiles: str, nBits=1024, radius=2) -> Optional[np.ndarray]:
    if (not HAVE_RDKIT) or (smiles is None) or (str(smiles).strip() == ""):
        return None
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return None
    bv = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.int8)
    DS.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)

def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    classes = np.unique(y)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(wi) for c, wi in zip(classes, w)}

def metrics_bin(y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    y_pred = (y_score >= thr).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_score)),
        "logloss": float(log_loss(y_true, np.clip(y_score, 1e-8, 1 - 1e-8))),
        "pos_rate": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }


# ============================================================
# Enzyme embeddings from H5
# ============================================================

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
    tok_norm = norm_key(last_token(enzyme_id))
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
            path = match_dataset_for_enzyme(str(eid), normbase_to_path, all_norm_bases)
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


# ============================================================
# Swap enumeration -> UNIQUE ProductName table
# ============================================================

CLS_ORDER = ["mono", "di", "tri"]

def split_products(s: str) -> List[str]:
    if pd.isna(s):
        return []
    parts = [p.strip() for p in str(s).split(";") if str(p).strip()]
    return parts if parts else []

def extract_hydroxyl_tokens_from_coretype(core_type: str) -> Tuple[str, ...]:
    """
    Parse Core-Type like '3a,7a,12a' -> ('3a','7a','12a')
    Ignore ketones 'k' tokens.
    """
    if core_type is None or str(core_type).strip() == "" or str(core_type).lower() == "nan":
        return tuple()
    s = str(core_type).lower().replace(" ", "")
    parts = [p for p in s.split(",") if p]
    toks = [p for p in parts if "k" not in p]
    toks = [re.sub(r"[^0-9ab]", "", t) for t in toks]  # keep digits + a/b
    toks = [t for t in toks if t]
    return tuple(sorted(set(toks)))

def hydroxyl_class_from_tokens(tokens: Tuple[str, ...]) -> str:
    n = len(tokens)
    if n <= 1: return "mono"
    if n == 2: return "di"
    return "tri"

def build_unique_product_table(enum_xlsx_path: str, amine_fp_bits: int, fp_radius: int) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Returns:
      prod: one row per unique ProductName with columns:
        ProductName, Amine_Name, Amine_SMILES, ba_class
      A_bits: amine feature matrix aligned to prod['amine_id'] mapping via amine_id
    """
    enum_df = pd.read_excel(enum_xlsx_path)

    for c in ["ProductName", "Amine_Name", "Amine_SMILES", "Core-Type"]:
        if c not in enum_df.columns:
            enum_df[c] = np.nan

    # explode ProductName lists into rows
    rows = []
    for _, r in enum_df.iterrows():
        plist = split_products(r.get("ProductName", np.nan))
        if not plist:
            pn = str(r.get("ProductName", "")).strip()
            plist = [pn] if pn else []
        for pn in plist:
            if not pn:
                continue
            rows.append({
                "ProductName": str(pn).strip(),
                "Amine_Name": r.get("Amine_Name", np.nan),
                "Amine_SMILES": r.get("Amine_SMILES", np.nan),
                "Core-Type": r.get("Core-Type", np.nan),
            })
    ex = pd.DataFrame(rows)
    ex = ex[ex["ProductName"].notna() & (ex["ProductName"].astype(str).str.strip() != "")]
    # IMPORTANT: "one row per ProductName" => pick a single representative row
    # Prefer non-null SMILES, else keep first.
    ex["has_smiles"] = ex["Amine_SMILES"].apply(lambda x: 0 if (x is None or str(x).strip()=="" or str(x).lower()=="nan") else 1)
    ex = ex.sort_values(["ProductName", "has_smiles"], ascending=[True, False])
    prod = ex.drop_duplicates(subset=["ProductName"], keep="first").copy()

    # derive BA class (mono/di/tri) from Core-Type hydroxyl tokens
    toks = prod["Core-Type"].apply(extract_hydroxyl_tokens_from_coretype)
    prod["ba_class"] = toks.apply(hydroxyl_class_from_tokens)

    # unique amines
    amines = (
        prod[["Amine_Name", "Amine_SMILES"]]
        .dropna(subset=["Amine_Name"])
        .drop_duplicates(subset=["Amine_Name"])
        .reset_index(drop=True)
    )
    amines["amine_id"] = np.arange(len(amines), dtype=int)
    name_to_id = dict(zip(amines["Amine_Name"].astype(str), amines["amine_id"].astype(int)))

    prod["amine_id"] = prod["Amine_Name"].astype(str).map(name_to_id)

    # build amine feature matrix (Morgan if possible else name-hash)
    A_bits = np.zeros((len(amines), amine_fp_bits), dtype=np.float32)
    n_morgan = 0
    for i, r in amines.iterrows():
        smi = r.get("Amine_SMILES", "")
        fp = morgan_fp(smi, nBits=amine_fp_bits, radius=fp_radius)
        if fp is None:
            fp = name_hash_fp(r.get("Amine_Name", "UNK"), nBits=amine_fp_bits)
        else:
            n_morgan += 1
        A_bits[i] = fp

    prod = prod.dropna(subset=["amine_id"]).reset_index(drop=True)

    meta = {
        "enum_xlsx": os.path.abspath(enum_xlsx_path),
        "unique_products": int(prod["ProductName"].nunique()),
        "unique_amines": int(len(amines)),
        "amine_fp_bits": int(amine_fp_bits),
        "fp_radius": int(fp_radius),
        "rdkit_available": bool(HAVE_RDKIT),
        "amine_morgan_count": int(n_morgan),
        "amine_hash_count": int(len(amines) - n_morgan),
    }
    return prod, A_bits, meta


# ============================================================
# Heatmap -> labeled (Enzyme, ProductName) pairs
# ============================================================

def build_pairs(heat_long: pd.DataFrame, prod: pd.DataFrame, enzyme_vecs: Dict[str, np.ndarray], thr: float) -> pd.DataFrame:
    """
    One row per (Enzyme, ProductName), label = max over replicates.
    Join on ProductName to get amine_id + ba_class.
    Drops rows missing enzyme embeddings or amine_id.
    """
    hl = heat_long.copy()
    hl["Enzyme"] = hl["Enzyme"].astype(str)
    hl["ProductName"] = hl["ProductName"].astype(str)

    if "Intensity" not in hl.columns and "label" not in hl.columns:
        raise ValueError("Heatmap must contain 'Intensity' or 'label' column.")

    if "label" in hl.columns and hl["label"].notna().any():
        hl["y"] = hl["label"].astype(float).astype(int)
    else:
        inten = pd.to_numeric(hl["Intensity"], errors="coerce").fillna(-1.0)
        hl["y"] = (inten > float(thr)).astype(int)

    # join product info (unique per ProductName)
    keep_cols = ["ProductName", "amine_id", "ba_class"]
    df = hl.merge(prod[keep_cols], on="ProductName", how="left")

    df["has_emb"] = df["Enzyme"].isin(enzyme_vecs.keys())
    df["has_am"] = df["amine_id"].notna()
    df = df[df["has_emb"] & df["has_am"]].copy()

    # collapse replicates: OR over replicate labels
    df = (
        df.groupby(["Enzyme", "ProductName", "amine_id", "ba_class"], as_index=False)
          .agg(y=("y", "max"))
    )
    return df


# ============================================================
# Enzyme holdout splitting (no leakage)
# ============================================================

def split_enzymes(enzymes: np.ndarray, train_frac: float, val_frac: float, test_frac: float, seed: int) -> Tuple[set, set, set]:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    uniq = np.unique(enzymes)
    rng.shuffle(uniq)
    n = len(uniq)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))
    enz_train = set(uniq[:n_train])
    enz_val = set(uniq[n_train:n_train+n_val])
    enz_test = set(uniq[n_train+n_val:])
    # sanity: disjoint
    assert enz_train.isdisjoint(enz_val)
    assert enz_train.isdisjoint(enz_test)
    assert enz_val.isdisjoint(enz_test)
    return enz_train, enz_val, enz_test


# ============================================================
# Feature building (Enzyme + Amine + BA class)
# ============================================================

def build_X(df_pairs: pd.DataFrame, enzyme_vecs: Dict[str, np.ndarray], A_bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    enzymes = df_pairs["Enzyme"].astype(str).tolist()
    amine_ids = df_pairs["amine_id"].astype(int).tolist()
    ba_cls = df_pairs["ba_class"].astype(str).tolist()
    y = df_pairs["y"].astype(int).values

    dE = next(iter(enzyme_vecs.values())).shape[0]
    dA = A_bits.shape[1]
    dC = len(CLS_ORDER)

    E = np.zeros((len(df_pairs), dE), dtype=np.float32)
    A = np.zeros((len(df_pairs), dA), dtype=np.float32)
    C = np.zeros((len(df_pairs), dC), dtype=np.float32)

    for i, e in enumerate(enzymes):
        E[i] = enzyme_vecs[e]
    for i, aid in enumerate(amine_ids):
        A[i] = A_bits[aid]
    for i, c in enumerate(ba_cls):
        c = c.lower().strip()
        if c not in CLS_ORDER:
            c = "mono"
        C[i, CLS_ORDER.index(c)] = 1.0

    X = np.concatenate([E, A, C], axis=1)
    meta = {
        "dims": {"enzyme": int(dE), "amine": int(dA), "class_onehot": int(dC), "X_total": int(X.shape[1])},
        "class_order": CLS_ORDER,
    }
    return X, y, meta


# ============================================================
# Model fits (XGB w/ learning curves, RF/MLP/LogReg)
# ============================================================

def fit_xgb_with_curve(Xtr, ytr, Xva, yva, seed: int) -> Tuple[xgb.Booster, float, pd.DataFrame]:
    """
    Use xgboost.train so early stopping always works across versions.
    """
    t0 = time.time()
    dtr = xgb.DMatrix(Xtr, label=ytr)
    dva = xgb.DMatrix(Xva, label=yva)

    # class imbalance
    neg = float((ytr == 0).sum())
    pos = float((ytr == 1).sum())
    spw = max(neg / max(pos, 1.0), 1.0)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "aucpr"],
        "eta": 0.03,
        "max_depth": 7,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "lambda": 1.0,
        "min_child_weight": 1.0,
        "scale_pos_weight": spw,
        "seed": seed,
        "tree_method": "hist",
    }

    evals_result = {}
    booster = xgb.train(
        params=params,
        dtrain=dtr,
        num_boost_round=4000,
        evals=[(dtr, "train"), (dva, "val")],
        early_stopping_rounds=150,
        evals_result=evals_result,
        verbose_eval=False,
    )
    dur = time.time() - t0

    # learning curve dataframe
    rounds = len(evals_result["train"]["logloss"])
    lc = pd.DataFrame({
        "round": np.arange(rounds),
        "train_logloss": evals_result["train"]["logloss"],
        "val_logloss": evals_result["val"]["logloss"],
        "train_aucpr": evals_result["train"]["aucpr"],
        "val_aucpr": evals_result["val"]["aucpr"],
    })
    return booster, dur, lc

def predict_xgb(booster: xgb.Booster, X) -> np.ndarray:
    d = xgb.DMatrix(X)
    return booster.predict(d, iteration_range=(0, booster.best_iteration + 1))

def fit_rf(Xtr, ytr, seed: int) -> Tuple[RandomForestClassifier, float]:
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=600,
        class_weight="balanced",
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(Xtr, ytr)
    return clf, time.time() - t0

def fit_logreg(Xtr, ytr, seed: int) -> Tuple[LogisticRegression, float]:
    t0 = time.time()
    cw = compute_class_weights(ytr)
    clf = LogisticRegression(max_iter=6000, class_weight=cw, random_state=seed)
    clf.fit(Xtr, ytr)
    return clf, time.time() - t0

def fit_mlp(Xtr, ytr, seed: int) -> Tuple[MLPClassifier, float]:
    t0 = time.time()
    clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=250,
        early_stopping=True,
        n_iter_no_change=15,
        random_state=seed,
    )
    clf.fit(Xtr, ytr)
    return clf, time.time() - t0


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    # paths
    heat_csv = pjoin(args.data_dir, args.heat_csv) if not os.path.isabs(args.heat_csv) else args.heat_csv
    h5_path = pjoin(args.data_dir, args.h5_path) if not os.path.isabs(args.h5_path) else args.h5_path
    enum_xlsx = pjoin(args.data_dir, args.enum_xlsx) if not os.path.isabs(args.enum_xlsx) else args.enum_xlsx

    out_root = pjoin(os.path.abspath(args.out_dir), args.run_tag)
    os.makedirs(out_root, exist_ok=True)

    print("== Config ==")
    print(f"heat_csv={heat_csv}")
    print(f"h5_path={h5_path}")
    print(f"enum_xlsx={enum_xlsx}")
    print(f"intensity_threshold={args.intensity_threshold}")
    print(f"holdout_repeats={args.holdout_repeats} train/val/test={args.train_frac}/{args.val_frac}/{args.test_frac}")
    print(f"RDKit available={HAVE_RDKIT}")
    print(f"outputs -> {out_root}")

    # load heatmap
    heat = pd.read_csv(heat_csv)
    if "Enzyme" not in heat.columns or "ProductName" not in heat.columns:
        raise ValueError("Heatmap must have columns: Enzyme, ProductName (and Intensity or label).")
    print(f"[Heat] rows={len(heat)} enzymes={heat['Enzyme'].nunique()} products={heat['ProductName'].nunique()}")

    # enzyme embeddings
    enzyme_ids = sorted(heat["Enzyme"].astype(str).unique().tolist())
    enzyme_vecs = load_enzyme_embeddings(h5_path, enzyme_ids)
    print(f"[Emb] loaded enzymes={len(enzyme_vecs)} dim={next(iter(enzyme_vecs.values())).shape[0]}")

    # unique ProductName -> single amine + class
    prod, A_bits, prod_meta = build_unique_product_table(
        enum_xlsx_path=enum_xlsx,
        amine_fp_bits=args.amine_fp_bits,
        fp_radius=args.fp_radius,
    )
    print(f"[Products] unique ProductName={prod['ProductName'].nunique()} | unique amines={A_bits.shape[0]} | A_bits={A_bits.shape}")
    prod.to_csv(pjoin(out_root, "unique_product_table.csv"), index=False)

    # join to build pairs
    pairs = build_pairs(heat, prod, enzyme_vecs, thr=args.intensity_threshold)
    print(f"[Pairs] rows={len(pairs)} enzymes={pairs['Enzyme'].nunique()} products={pairs['ProductName'].nunique()}")
    print(f"[Labels] pos={int((pairs['y']==1).sum())} neg={int((pairs['y']==0).sum())} pos%={pairs['y'].mean():.3f}")

    # build X
    X_raw, y, feat_meta = build_X(pairs, enzyme_vecs, A_bits)
    enzymes = pairs["Enzyme"].astype(str).values

    # scale (for MLP/logreg)
    scaler = StandardScaler().fit(X_raw)
    X_scaled = scaler.transform(X_raw).astype(np.float32)

    feature_meta = {
        "run_tag": args.run_tag,
        "intensity_threshold": float(args.intensity_threshold),
        "product_meta": prod_meta,
        "feature_meta": feat_meta,
        "rdkit_available": bool(HAVE_RDKIT),
        "n_rows": int(len(pairs)),
        "n_enzymes": int(pairs["Enzyme"].nunique()),
        "n_products": int(pairs["ProductName"].nunique()),
        "label_pos_rate": float(np.mean(y)),
    }
    with open(pjoin(out_root, "feature_meta.json"), "w") as f:
        json.dump(feature_meta, f, indent=2)

    # models enabled
    models = []
    if not args.no_xgb: models.append("xgb")
    if not args.no_rf: models.append("rf")
    if not args.no_mlp: models.append("mlp")
    if not args.no_logreg: models.append("logreg")
    print(f"[Models] {models}")

    metrics_rows = []
    preds_rows = []

    # repeated enzyme holdout
    for rep in range(1, args.holdout_repeats + 1):
        rep_seed = args.seed + 1000 * rep
        enz_train, enz_val, enz_test = split_enzymes(
            enzymes=enzymes,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=rep_seed
        )

        tr_mask = np.array([e in enz_train for e in enzymes], dtype=bool)
        va_mask = np.array([e in enz_val for e in enzymes], dtype=bool)
        te_mask = np.array([e in enz_test for e in enzymes], dtype=bool)

        # leakage sanity check
        assert not (set(enzymes[tr_mask]) & set(enzymes[va_mask]))
        assert not (set(enzymes[tr_mask]) & set(enzymes[te_mask]))
        assert not (set(enzymes[va_mask]) & set(enzymes[te_mask]))

        print(f"\n== Holdout repeat {rep}/{args.holdout_repeats} ==")
        print(f"  rows train/val/test = {tr_mask.sum()}/{va_mask.sum()}/{te_mask.sum()}")
        print(f"  pos% train/val/test = {y[tr_mask].mean():.3f}/{y[va_mask].mean():.3f}/{y[te_mask].mean():.3f}")
        print(f"  enzymes train/val/test = {len(enz_train)}/{len(enz_val)}/{len(enz_test)}")

        # choose X matrices
        Xtr_raw, Xva_raw, Xte_raw = X_raw[tr_mask], X_raw[va_mask], X_raw[te_mask]
        Xtr_s, Xva_s, Xte_s = X_scaled[tr_mask], X_scaled[va_mask], X_scaled[te_mask]
        ytr, yva, yte = y[tr_mask], y[va_mask], y[te_mask]

        # --- XGB ---
        if "xgb" in models:
            booster, tr_time, lc = fit_xgb_with_curve(Xtr_raw, ytr, Xva_raw, yva, seed=rep_seed)
            lc_path = pjoin(out_root, f"xgb_holdout_rep{rep}_learning_curve.csv")
            lc.to_csv(lc_path, index=False)

            p_val = predict_xgb(booster, Xva_raw)
            p_test = predict_xgb(booster, Xte_raw)

            m_val = metrics_bin(yva, p_val)
            m_test = metrics_bin(yte, p_test)

            metrics_rows.append({
                "rep": rep, "model": "xgb", "train_time_sec": tr_time,
                **{f"val_{k}": v for k, v in m_val.items()},
                **{f"test_{k}": v for k, v in m_test.items()},
                "best_iteration": int(booster.best_iteration),
            })
            preds_rows.append(pd.DataFrame({
                "rep": rep, "model": "xgb",
                "y_true": yte, "y_score": p_test
            }))

        # --- RF ---
        if "rf" in models:
            rf, tr_time = fit_rf(Xtr_raw, ytr, seed=rep_seed)
            p_val = rf.predict_proba(Xva_raw)[:, 1]
            p_test = rf.predict_proba(Xte_raw)[:, 1]
            m_val = metrics_bin(yva, p_val)
            m_test = metrics_bin(yte, p_test)
            metrics_rows.append({
                "rep": rep, "model": "rf", "train_time_sec": tr_time,
                **{f"val_{k}": v for k, v in m_val.items()},
                **{f"test_{k}": v for k, v in m_test.items()},
            })
            preds_rows.append(pd.DataFrame({
                "rep": rep, "model": "rf",
                "y_true": yte, "y_score": p_test
            }))

        # --- MLP ---
        if "mlp" in models:
            mlp, tr_time = fit_mlp(Xtr_s, ytr, seed=rep_seed)
            p_val = mlp.predict_proba(Xva_s)[:, 1]
            p_test = mlp.predict_proba(Xte_s)[:, 1]
            m_val = metrics_bin(yva, p_val)
            m_test = metrics_bin(yte, p_test)
            metrics_rows.append({
                "rep": rep, "model": "mlp", "train_time_sec": tr_time,
                **{f"val_{k}": v for k, v in m_val.items()},
                **{f"test_{k}": v for k, v in m_test.items()},
            })
            preds_rows.append(pd.DataFrame({
                "rep": rep, "model": "mlp",
                "y_true": yte, "y_score": p_test
            }))

        # --- LogReg ---
        if "logreg" in models:
            lr, tr_time = fit_logreg(Xtr_s, ytr, seed=rep_seed)
            p_val = lr.predict_proba(Xva_s)[:, 1]
            p_test = lr.predict_proba(Xte_s)[:, 1]
            m_val = metrics_bin(yva, p_val)
            m_test = metrics_bin(yte, p_test)
            metrics_rows.append({
                "rep": rep, "model": "logreg", "train_time_sec": tr_time,
                **{f"val_{k}": v for k, v in m_val.items()},
                **{f"test_{k}": v for k, v in m_test.items()},
            })
            preds_rows.append(pd.DataFrame({
                "rep": rep, "model": "logreg",
                "y_true": yte, "y_score": p_test
            }))

    # Save metrics + predictions
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = pjoin(out_root, "metrics_per_repeat.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # summary across repeats (mean/std on TEST)
    summary_rows = []
    for model_name in sorted(metrics_df["model"].unique()):
        m = metrics_df[metrics_df["model"] == model_name]
        row = {"model": model_name, "repeats": int(len(m))}
        for col in ["test_pr_auc", "test_roc_auc", "test_f1", "test_mcc", "test_acc", "test_brier", "test_logloss"]:
            row[f"{col}_mean"] = float(m[col].mean())
            row[f"{col}_std"] = float(m[col].std(ddof=0))
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = pjoin(out_root, "metrics_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    preds_df = pd.concat(preds_rows, ignore_index=True) if preds_rows else pd.DataFrame()
    preds_path = pjoin(out_root, "predictions_test.csv")
    preds_df.to_csv(preds_path, index=False)

    run_meta = {
        "args": vars(args),
        "outputs": {
            "unique_product_table.csv": pjoin(out_root, "unique_product_table.csv"),
            "feature_meta.json": pjoin(out_root, "feature_meta.json"),
            "metrics_per_repeat.csv": metrics_path,
            "metrics_summary.csv": summary_path,
            "predictions_test.csv": preds_path,
            "xgb_learning_curves": "xgb_holdout_rep*_learning_curve.csv",
        }
    }
    with open(pjoin(out_root, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    print("\n=== Saved ===")
    print(f"[Saved] {pjoin(out_root, 'unique_product_table.csv')}")
    print(f"[Saved] {pjoin(out_root, 'feature_meta.json')}")
    print(f"[Saved] {metrics_path}")
    print(f"[Saved] {summary_path}")
    print(f"[Saved] {preds_path}")
    if "xgb" in models:
        print(f"[Saved] {pjoin(out_root, 'xgb_holdout_rep*_learning_curve.csv')}")
    print(f"[Saved] {pjoin(out_root, 'run_meta.json')}")
    print(f"\nDone. Root outputs: {out_root}")


if __name__ == "__main__":
    main()
