import os, re, json, time, argparse, warnings, hashlib
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, accuracy_score,
    matthews_corrcoef, brier_score_loss, log_loss
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ---- XGBoost required ----
from xgboost import XGBClassifier

# ---- RDKit optional; fallback to name-hash fingerprints if missing ----
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs as DS
    HAVE_RDKIT = True
except Exception:
    HAVE_RDKIT = False

import h5py


# ============================================================
# Args
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser("Baseline enzyme+amine activity classifier with enzyme-holdout splits")

    ap.add_argument("--data_dir", type=str, default="/home/adsiordia/BSH-Model")
    ap.add_argument("--out_dir",  type=str, default="/home/adsiordia/BSH-Model/outputs")
    ap.add_argument("--run_tag",  type=str, default="baseline_enzyme_amine")

    ap.add_argument("--heat_csv", type=str, default="ipsita_heatmap_long.csv")
    ap.add_argument("--h5_path",  type=str, default="Seqs_list_total.h5")
    ap.add_argument("--enum_xlsx", type=str, default="swap_enumeration_with_core_smiles.xlsx")  # contains ProductName, Amine_Name, Amine_SMILES

    ap.add_argument("--intensity_threshold", type=float, default=1000.0)

    ap.add_argument("--holdout_repeats", type=int, default=5)
    ap.add_argument("--train_frac", type=float, default=0.70)
    ap.add_argument("--val_frac",   type=float, default=0.15)
    ap.add_argument("--test_frac",  type=float, default=0.15)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_split_tries", type=int, default=200)
    ap.add_argument("--posrate_tol", type=float, default=0.08, help="acceptable abs diff in pos-rate vs global per split")

    ap.add_argument("--amine_fp_bits", type=int, default=1024)
    ap.add_argument("--fp_radius", type=int, default=2)

    # XGB training curve
    ap.add_argument("--xgb_rounds", type=int, default=500)
    ap.add_argument("--xgb_max_depth", type=int, default=7)
    ap.add_argument("--xgb_lr", type=float, default=0.05)

    return ap.parse_args()


# ============================================================
# Helpers
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
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16)

def compute_class_weights_dict(y: np.ndarray) -> Dict[int, float]:
    classes = np.unique(y)
    w = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(wi) for c, wi in zip(classes, w)}

def metrics_bin(y_true: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    y_hat = (p >= 0.5).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, p)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "f1": float(f1_score(y_true, y_hat)),
        "mcc": float(matthews_corrcoef(y_true, y_hat)),
        "acc": float(accuracy_score(y_true, y_hat)),
        "brier": float(brier_score_loss(y_true, p)),
        "logloss": float(log_loss(y_true, np.clip(p, 1e-8, 1 - 1e-8))),
        "pos_rate": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }

def split_products(s: str) -> List[str]:
    if pd.isna(s): return []
    parts = [p.strip() for p in str(s).split(";") if str(p).strip()]
    return parts if parts else []


# ============================================================
# Fingerprints
# ============================================================

def morgan_fp(smiles: str, nBits=1024, radius=2) -> np.ndarray:
    out = np.zeros((nBits,), dtype=np.float32)
    if (not HAVE_RDKIT) or (smiles is None) or (str(smiles).strip() == ""):
        return out
    m = Chem.MolFromSmiles(str(smiles))
    if m is None:
        return out
    bv = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.int8)
    DS.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)

def name_hash_fp(name: str, nBits=1024) -> np.ndarray:
    """Fallback if SMILES missing/invalid: hashed 3-grams of name."""
    s = (str(name) or "").lower().strip()
    grams = set(); t = f"^{s}$"
    for i in range(max(0, len(t)-2)):
        grams.add(t[i:i+3])
    bits = np.zeros((nBits,), dtype=np.float32)
    for g in grams:
        idx = stable_hash_int(g) % nBits
        bits[idx] = 1.0
    return bits


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


# ============================================================
# Build ProductName -> amine_id and amine fingerprints
# ============================================================

def build_membership_and_amine_fps(enum_xlsx_path: str,
                                   fp_bits: int,
                                   fp_radius: int) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    membership: ProductName -> amine_id (based on Amine_Name)
    amines_df: unique amines with amine_id
    A_bits: [n_amines, fp_bits] amine fingerprints
    """
    df = pd.read_excel(enum_xlsx_path)

    for c in ["ProductName", "Amine_Name", "Amine_SMILES"]:
        if c not in df.columns:
            df[c] = np.nan

    # explode ProductName lists
    rows = []
    for _, r in df.iterrows():
        pnames = split_products(r["ProductName"])
        if not pnames:
            pn = str(r["ProductName"]).strip()
            pnames = [pn] if pn else []
        am_name = str(r.get("Amine_Name", "") or "").strip()
        am_smi  = str(r.get("Amine_SMILES", "") or "").strip()
        if not am_name:
            continue
        for pn in pnames:
            if not pn:
                continue
            rows.append({"ProductName": pn, "Amine_Name": am_name, "Amine_SMILES": am_smi})

    membership = pd.DataFrame(rows).drop_duplicates(subset=["ProductName"]).reset_index(drop=True)

    # unique amines
    amines = (membership[["Amine_Name", "Amine_SMILES"]]
              .drop_duplicates(subset=["Amine_Name"])
              .reset_index(drop=True))
    amines["amine_id"] = np.arange(len(amines), dtype=int)

    membership = membership.merge(amines[["Amine_Name", "amine_id"]], on="Amine_Name", how="left")

    # build fingerprints (prefer SMILES; fallback to name hash)
    A_bits = np.zeros((len(amines), fp_bits), dtype=np.float32)
    n_smi = n_hash = 0
    for i, r in amines.iterrows():
        smi = str(r.get("Amine_SMILES", "") or "").strip()
        nm  = str(r.get("Amine_Name", "") or "").strip()
        fp = morgan_fp(smi, nBits=fp_bits, radius=fp_radius)
        if fp.sum() == 0.0:
            fp = name_hash_fp(nm if nm else "UNK_AMINE", nBits=fp_bits)
            n_hash += 1
        else:
            n_smi += 1
        A_bits[i] = fp

    print(f"[Amines] unique={len(amines)} | RDKit={HAVE_RDKIT} | SMILES_FPs={n_smi} | name-hash={n_hash}")
    return membership, amines, A_bits


# ============================================================
# Heatmap -> labeled pairs (enzyme, product -> amine)
# ============================================================

def derive_label_from_intensity(df: pd.DataFrame, thr: float) -> np.ndarray:
    if "label" in df.columns and df["label"].notna().any():
        return df["label"].astype(float).astype(int).values
    if "Intensity" not in df.columns:
        raise ValueError("Heatmap must have 'Intensity' or 'label'.")
    inten = pd.to_numeric(df["Intensity"], errors="coerce").fillna(-1.0)
    return (inten > float(thr)).astype(int).values

def make_pairs_enzyme_amine(heat_long: pd.DataFrame,
                            membership: pd.DataFrame,
                            enzyme_vecs: Dict[str, np.ndarray],
                            thr: float) -> pd.DataFrame:
    """
    Join heatmap with membership on ProductName -> amine_id.
    Collapse replicates: label = max over replicates for same (Enzyme, amine_id).
    """
    hl = heat_long.copy()
    hl["Enzyme"] = hl["Enzyme"].astype(str)
    hl["ProductName"] = hl["ProductName"].astype(str)

    df = hl.merge(membership[["ProductName", "amine_id"]], on="ProductName", how="left")
    df["y"] = derive_label_from_intensity(df, thr)
    df["has_emb"] = df["Enzyme"].isin(enzyme_vecs.keys())
    df["has_am"]  = df["amine_id"].notna()
    df = df[df["has_emb"] & df["has_am"]].copy()

    # collapse to enzyme+amine (this is key for your baseline)
    df = (df.groupby(["Enzyme", "amine_id"], as_index=False)
            .agg(y=("y", "max")))  # any active replicate -> active
    return df


# ============================================================
# Enzyme-level holdout splits (retry until pos-rate is reasonable)
# ============================================================

def enzyme_holdout_split(pairs: pd.DataFrame,
                         train_frac: float, val_frac: float, test_frac: float,
                         seed: int, rep: int,
                         max_tries: int, posrate_tol: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    rng = np.random.default_rng(seed + 1000 * rep)
    enzymes = pairs["Enzyme"].astype(str).unique().tolist()
    global_pos = pairs["y"].mean()

    # compute per-enzyme counts and pos
    perE = (pairs.groupby("Enzyme")["y"]
            .agg(["mean", "count"])
            .rename(columns={"mean":"pos_rate","count":"n"}))
    enz_list = perE.index.tolist()

    nE = len(enz_list)
    n_train = max(1, int(round(train_frac * nE)))
    n_val   = max(1, int(round(val_frac   * nE)))
    n_test  = max(1, nE - n_train - n_val)

    best = None
    for t in range(max_tries):
        rng.shuffle(enz_list)
        trE = set(enz_list[:n_train])
        vaE = set(enz_list[n_train:n_train+n_val])
        teE = set(enz_list[n_train+n_val:n_train+n_val+n_test])

        tr = pairs["Enzyme"].isin(trE).values
        va = pairs["Enzyme"].isin(vaE).values
        te = pairs["Enzyme"].isin(teE).values

        if tr.sum() == 0 or va.sum() == 0 or te.sum() == 0:
            continue

        pr_tr = pairs.loc[tr, "y"].mean()
        pr_va = pairs.loc[va, "y"].mean()
        pr_te = pairs.loc[te, "y"].mean()

        ok = (abs(pr_tr - global_pos) <= posrate_tol and
              abs(pr_va - global_pos) <= posrate_tol and
              abs(pr_te - global_pos) <= posrate_tol)

        score = abs(pr_tr-global_pos) + abs(pr_va-global_pos) + abs(pr_te-global_pos)
        if best is None or score < best["score"]:
            best = {"tr": tr, "va": va, "te": te, "score": score,
                    "rates": (float(pr_tr), float(pr_va), float(pr_te)),
                    "enz_counts": (len(trE), len(vaE), len(teE))}
        if ok:
            break

    meta = {
        "global_pos_rate": float(global_pos),
        "train_pos_rate": best["rates"][0],
        "val_pos_rate": best["rates"][1],
        "test_pos_rate": best["rates"][2],
        "enz_counts": {"train": best["enz_counts"][0], "val": best["enz_counts"][1], "test": best["enz_counts"][2]},
        "tries_used": int(t+1),
        "posrate_tol": float(posrate_tol),
    }
    return best["tr"], best["va"], best["te"], meta


# ============================================================
# Feature assembly: enzyme emb + amine fp
# ============================================================

def build_X(pairs: pd.DataFrame, enzyme_vecs: Dict[str, np.ndarray], A_bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    enzymes = pairs["Enzyme"].astype(str).tolist()
    amine_ids = pairs["amine_id"].astype(int).tolist()
    y = pairs["y"].astype(int).values
    groups = pairs["Enzyme"].astype(str).values

    dE = next(iter(enzyme_vecs.values())).shape[0]
    dA = A_bits.shape[1]
    E = np.zeros((len(pairs), dE), dtype=np.float32)
    A = np.zeros((len(pairs), dA), dtype=np.float32)

    for i, (e, aid) in enumerate(zip(enzymes, amine_ids)):
        E[i] = enzyme_vecs[e]
        A[i] = A_bits[aid]

    X = np.concatenate([E, A], axis=1)
    return X, y, groups


# ============================================================
# Model fits + curve logging
# ============================================================

def fit_xgb_with_curve(Xtr, ytr, Xva, yva, seed: int,
                       rounds: int, max_depth: int, lr: float) -> Tuple[XGBClassifier, dict, float]:
    neg = float((ytr == 0).sum())
    pos = float((ytr == 1).sum())
    spw = max(neg / max(pos, 1.0), 1.0)

    clf = XGBClassifier(
        n_estimators=rounds,
        max_depth=max_depth,
        learning_rate=lr,
        subsample=0.9,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        objective="binary:logistic",
        tree_method="hist",
        max_bin=256,
        eval_metric=["logloss", "aucpr"],
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
    dt = time.time() - t0
    hist = clf.evals_result()
    return clf, hist, dt


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    root_out = pjoin(os.path.abspath(args.out_dir), args.run_tag)
    os.makedirs(root_out, exist_ok=True)

    heat_csv_path = pjoin(args.data_dir, args.heat_csv) if not os.path.isabs(args.heat_csv) else args.heat_csv
    h5_path       = pjoin(args.data_dir, args.h5_path)  if not os.path.isabs(args.h5_path) else args.h5_path
    enum_path     = pjoin(args.data_dir, args.enum_xlsx) if not os.path.isabs(args.enum_xlsx) else args.enum_xlsx

    print("== Config ==")
    print(f"heat_csv={heat_csv_path}")
    print(f"h5_path={h5_path}")
    print(f"enum_xlsx={enum_path}")
    print(f"intensity_threshold={args.intensity_threshold}")
    print(f"holdout_repeats={args.holdout_repeats} (train/val/test={args.train_frac}/{args.val_frac}/{args.test_frac})")
    print(f"RDKit available={HAVE_RDKIT}")

    # Load heatmap
    heat_long = pd.read_csv(heat_csv_path)
    if "Enzyme" not in heat_long.columns or "ProductName" not in heat_long.columns:
        raise ValueError("Heatmap long CSV must contain 'Enzyme' and 'ProductName' columns.")
    print(f"[Heatmap] rows={len(heat_long)} enzymes={heat_long['Enzyme'].nunique()} products={heat_long['ProductName'].nunique()}")

    # Load enzyme embeddings
    enzyme_ids = sorted(heat_long["Enzyme"].astype(str).unique().tolist())
    enzyme_vecs = load_enzyme_embeddings(h5_path, enzyme_ids)
    dE = next(iter(enzyme_vecs.values())).shape[0]
    print(f"[Embeddings] loaded enzymes={len(enzyme_vecs)} dim={dE}")

    # Build membership + amine fingerprints
    membership, amines_df, A_bits = build_membership_and_amine_fps(
        enum_xlsx_path=enum_path,
        fp_bits=args.amine_fp_bits,
        fp_radius=args.fp_radius
    )

    # Build enzyme+amine labeled pairs
    pairs = make_pairs_enzyme_amine(
        heat_long=heat_long,
        membership=membership,
        enzyme_vecs=enzyme_vecs,
        thr=args.intensity_threshold
    )
    print(f"[Pairs enzyme+amine] rows={len(pairs)} enzymes={pairs['Enzyme'].nunique()} unique_amines={pairs['amine_id'].nunique()}")
    print(f"[Label balance] pos={int(pairs['y'].sum())} ({100*pairs['y'].mean():.1f}%) neg={int((pairs['y']==0).sum())}")

    # Assemble full X
    X, y, groups = build_X(pairs, enzyme_vecs, A_bits)

    # Save a feature meta snapshot
    feature_meta = {
        "task": "binary activity",
        "features": ["enzyme_embedding", "amine_fingerprint"],
        "dims": {"enzyme_emb": int(dE), "amine_fp": int(A_bits.shape[1]), "X_total": int(X.shape[1])},
        "intensity_threshold": float(args.intensity_threshold),
        "rdkit_available": HAVE_RDKIT,
        "n_pairs": int(len(pairs)),
        "n_enzymes": int(pairs["Enzyme"].nunique()),
        "n_amines": int(pairs["amine_id"].nunique()),
    }
    with open(pjoin(root_out, "feature_meta.json"), "w") as f:
        json.dump(feature_meta, f, indent=2)

    metrics_rows = []
    pred_rows = []
    split_rows = []

    for rep in range(1, args.holdout_repeats + 1):
        print(f"\n== Holdout repeat {rep}/{args.holdout_repeats} ==")

        tr_mask, va_mask, te_mask, split_meta = enzyme_holdout_split(
            pairs=pairs,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
            rep=rep,
            max_tries=args.max_split_tries,
            posrate_tol=args.posrate_tol
        )

        Xtr, ytr = X[tr_mask], y[tr_mask]
        Xva, yva = X[va_mask], y[va_mask]
        Xte, yte = X[te_mask], y[te_mask]

        print(f"  rows train/val/test = {len(ytr)}/{len(yva)}/{len(yte)}")
        print(f"  pos% train/val/test = {ytr.mean():.3f}/{yva.mean():.3f}/{yte.mean():.3f} (global={y.mean():.3f})")
        print(f"  enzyme counts train/val/test = {split_meta['enz_counts']} (tries_used={split_meta['tries_used']})")

        split_meta_row = {"rep": rep, **split_meta}
        split_rows.append(split_meta_row)

        # Scaling for linear/MLP only
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr).astype(np.float32)
        Xva_s = scaler.transform(Xva).astype(np.float32)
        Xte_s = scaler.transform(Xte).astype(np.float32)

        cw = compute_class_weights_dict(ytr)

        # --- Models ---
        models = {}

        # XGB (raw)
        xgb_model, xgb_hist, xgb_time = fit_xgb_with_curve(
            Xtr, ytr, Xva, yva,
            seed=args.seed + rep,
            rounds=args.xgb_rounds,
            max_depth=args.xgb_max_depth,
            lr=args.xgb_lr
        )
        models["xgb"] = ("raw", xgb_model, xgb_time, xgb_hist)

        # RF (raw)
        t0 = time.time()
        rf = RandomForestClassifier(
            n_estimators=600,
            class_weight="balanced",
            n_jobs=-1,
            random_state=args.seed + rep
        )
        rf.fit(Xtr, ytr)
        rf_time = time.time() - t0
        models["rf"] = ("raw", rf, rf_time, None)

        # MLP (scaled)
        t0 = time.time()
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=250,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=args.seed + rep
        )
        mlp.fit(Xtr_s, ytr)
        mlp_time = time.time() - t0
        models["mlp"] = ("scaled", mlp, mlp_time, {"mlp_train_loss": getattr(mlp, "loss_curve_", None)})

        # Logistic regression (scaled) – sanity baseline
        t0 = time.time()
        lr = LogisticRegression(max_iter=8000, class_weight=cw)
        lr.fit(Xtr_s, ytr)
        lr_time = time.time() - t0
        models["logreg"] = ("scaled", lr, lr_time, None)

        # Evaluate each model
        for name, (xtype, model, train_time, hist) in models.items():
            if xtype == "raw":
                p_val = model.predict_proba(Xva)[:, 1]
                p_tst = model.predict_proba(Xte)[:, 1]
            else:
                p_val = model.predict_proba(Xva_s)[:, 1]
                p_tst = model.predict_proba(Xte_s)[:, 1]

            m_val = metrics_bin(yva, p_val)
            m_tst = metrics_bin(yte, p_tst)

            metrics_rows.append({
                "rep": rep,
                "model": name,
                "train_time_sec": float(train_time),

                "val_pr_auc": m_val["pr_auc"],
                "val_roc_auc": m_val["roc_auc"],
                "val_f1": m_val["f1"],
                "val_mcc": m_val["mcc"],
                "val_acc": m_val["acc"],
                "val_brier": m_val["brier"],
                "val_logloss": m_val["logloss"],
                "val_pos_rate": m_val["pos_rate"],
                "val_n": m_val["n"],

                "test_pr_auc": m_tst["pr_auc"],
                "test_roc_auc": m_tst["roc_auc"],
                "test_f1": m_tst["f1"],
                "test_mcc": m_tst["mcc"],
                "test_acc": m_tst["acc"],
                "test_brier": m_tst["brier"],
                "test_logloss": m_tst["logloss"],
                "test_pos_rate": m_tst["pos_rate"],
                "test_n": m_tst["n"],
            })

            pred_rows.append(pd.DataFrame({
                "rep": rep,
                "model": name,
                "y_true": yte,
                "y_score": p_tst,
            }))

            # Save learning curves per repeat (XGB)
            if name == "xgb" and hist is not None:
                lc = pd.DataFrame({
                    "round": np.arange(len(hist["validation_0"]["logloss"])),
                    "train_logloss": hist["validation_0"]["logloss"],
                    "val_logloss": hist["validation_1"]["logloss"],
                    "train_aucpr": hist["validation_0"]["aucpr"],
                    "val_aucpr": hist["validation_1"]["aucpr"],
                })
                lc.to_csv(pjoin(root_out, f"holdout_rep{rep}_learning_curve.csv"), index=False)

            # Save MLP loss curve per repeat (training only)
            if name == "mlp" and hist is not None:
                lc = hist.get("mlp_train_loss", None)
                if lc is not None:
                    pd.DataFrame({
                        "epoch": np.arange(len(lc)),
                        "train_loss": lc
                    }).to_csv(pjoin(root_out, f"holdout_rep{rep}_mlp_train_loss.csv"), index=False)

    # Save metrics + predictions + splits
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(pjoin(root_out, "metrics_per_repeat.csv"), index=False)

    preds_df = pd.concat(pred_rows, ignore_index=True)
    preds_df.to_csv(pjoin(root_out, "predictions_test.csv"), index=False)

    pd.DataFrame(split_rows).to_csv(pjoin(root_out, "split_summary.csv"), index=False)

    # Summary table
    summary = (metrics_df.groupby("model")[[
        "val_pr_auc","val_roc_auc","val_f1","val_mcc","val_acc","val_brier","val_logloss",
        "test_pr_auc","test_roc_auc","test_f1","test_mcc","test_acc","test_brier","test_logloss"
    ]]
    .agg(["mean","std"])
    .reset_index())
    summary.to_csv(pjoin(root_out, "metrics_summary.csv"), index=False)

    # Run meta
    run_meta = {
        "root_out": root_out,
        "args": vars(args),
        "feature_meta": feature_meta,
    }
    with open(pjoin(root_out, "run_meta.json"), "w") as f:
        json.dump(run_meta, f, indent=2)

    print("\n=== Saved ===")
    print(f"[Saved] {pjoin(root_out, 'feature_meta.json')}")
    print(f"[Saved] {pjoin(root_out, 'metrics_per_repeat.csv')}")
    print(f"[Saved] {pjoin(root_out, 'metrics_summary.csv')}")
    print(f"[Saved] {pjoin(root_out, 'predictions_test.csv')}")
    print(f"[Saved] {pjoin(root_out, 'split_summary.csv')}")
    print(f"[Saved] {pjoin(root_out, 'run_meta.json')}")
    print("\nDone. Root outputs:", root_out)


if __name__ == "__main__":
    main()
