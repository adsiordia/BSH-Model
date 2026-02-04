import os, re, gc, json, time, math, argparse, warnings, csv
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle as sk_shuffle

warnings.filterwarnings("ignore", category=UserWarning)

# ---- XGBoost REQUIRED ----
try:
    from xgboost import XGBClassifier
except Exception as e:
    raise SystemExit("xgboost is required. Install with: pip install xgboost\n" + str(e))

# ---- RDKit OPTIONAL ----
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import DataStructs as DS
    HAVE_RDKIT = True
except Exception:
    HAVE_RDKIT = False


# =================== CLI ===================
def parse_args():
    ap = argparse.ArgumentParser(description="Train pooled-bag classifier (amine keyed by name) + pred logs")
    ap.add_argument("--data_dir", type=str, default="/home/adsiordia/BSH-Model")
    ap.add_argument("--out_dir",  type=str, default="/home/adsiordia/BSH-Model/outputs")
    ap.add_argument("--run-tag",  type=str, default="")
    ap.add_argument("--splits",   type=int, default=5, help="GroupKFold splits per repeat")
    ap.add_argument("--repeats",  type=int, default=3, help="CV repeats with different enzyme permutations")
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--intensity-threshold", type=float, default=0.0,
                    help="If 'label' missing: label = (Intensity > threshold)")
    ap.add_argument("--no-rf", action="store_true", help="Disable RandomForest")
    ap.add_argument("--et", action="store_true", help="Enable ExtraTrees")
    ap.add_argument("--calib-cv", type=int, default=3, help="Folds for probability calibration")
    ap.add_argument("--save-cache", action="store_true", help="Save X/y/groups cache")
    ap.add_argument("--load-cache", action="store_true", help="Load X/y/groups cache if present")

    # Optional fixed enzyme hold-out
    ap.add_argument("--holdout", type=str, default="",
                    help="Optional enzyme-level train,val,test fractions, e.g. '0.7,0.15,0.15'")
    ap.add_argument("--holdout-repeats", type=int, default=1, help="Repeat holdout with different enzyme partitions")
    return ap.parse_args()


# =================== Config ===================
@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str
    n_splits: int
    n_repeats: int
    seed: int
    intensity_threshold: float
    use_rf: bool
    use_et: bool
    calib_cv: int
    save_cache: bool
    load_cache: bool


# =================== Helpers ===================
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)

def pjoin(*a): return os.path.join(*a)

def norm_key(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()

def last_token(code: str) -> str:
    return str(code).split("_")[-1]

def _append_row(csv_path, fieldnames, row):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


# =================== Enzyme embeddings from H5 ===================
import h5py

def collect_h5_datasets(h5_path):
    paths = []
    with h5py.File(h5_path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                paths.append(name)
        f.visititems(visit)
    return paths

def match_dataset_for_enzyme(enzyme_id: str, normbase_to_path: Dict[str,str], all_norm_bases):
    full_norm = norm_key(enzyme_id)
    tok_norm  = norm_key(last_token(enzyme_id))
    if full_norm in normbase_to_path: return normbase_to_path[full_norm]
    if tok_norm  in normbase_to_path: return normbase_to_path[tok_norm]
    hits = [b for b in all_norm_bases if tok_norm in b or b in tok_norm]
    if hits: return normbase_to_path[sorted(hits, key=len)[0]]
    from difflib import get_close_matches
    cand = get_close_matches(tok_norm, all_norm_bases, n=1, cutoff=0.92)
    return normbase_to_path[cand[0]] if cand else None

def load_or_build_enzyme_embeddings(h5_path: str, heat_long_csv: str, out_dir: str) -> Dict[str, np.ndarray]:
    npy_path = pjoin(out_dir, "enzyme_embeddings.npy")
    if os.path.exists(npy_path):
        return np.load(npy_path, allow_pickle=True).item()

    heat_long = pd.read_csv(heat_long_csv)
    enz_ids = sorted(heat_long["Enzyme"].astype(str).unique().tolist())

    all_paths = collect_h5_datasets(h5_path)
    if not all_paths: raise RuntimeError(f"No datasets found in {h5_path}.")
    bases = [p.rsplit("/", 1)[-1] for p in all_paths]
    norm_bases = [norm_key(b) for b in bases]
    normbase_to_path = {}
    for base, normb, full in zip(bases, norm_bases, all_paths):
        normbase_to_path.setdefault(normb, full)
    all_norm_bases = list(normbase_to_path.keys())

    rows = []
    for eid in tqdm(enz_ids, desc="Embedding enzymes", unit="enz"):
        path = match_dataset_for_enzyme(eid, normbase_to_path, all_norm_bases)
        if path is None: continue
        with h5py.File(h5_path, "r") as f:
            arr = np.array(f[path])
        if arr.ndim == 2 and arr.shape[0] > 1: vec = arr.mean(axis=0).astype(np.float32)
        elif arr.ndim == 2 and arr.shape[0] == 1: vec = arr[0].astype(np.float32)
        elif arr.ndim > 2: vec = arr.reshape(arr.shape[-1]).astype(np.float32)
        else: vec = arr.astype(np.float32)
        rows.append((eid, vec))
    if not rows: raise RuntimeError("No embeddings loaded—check H5 structure and enzyme naming.")
    data = {eid: vec for (eid, vec) in rows}
    np.save(npy_path, data)
    return data


# =================== SWAP enum → ProductName→Amine_Name + AMINE bits ===================
def split_products(s: str):
    if pd.isna(s): return []
    parts = [p.strip() for p in str(s).split(";") if str(p).strip()]
    return parts if parts else []

def morgan_fp(smi, nBits=1024, radius=2):
    if not smi or not HAVE_RDKIT: return None
    m = Chem.MolFromSmiles(str(smi))
    if m is None: return None
    bv = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.int8)
    DS.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)

def name_hash_fp(name: str, nBits=1024):
    s = (str(name) or "").lower().strip()
    grams = set(); t = f"^{s}$"
    for i in range(max(0, len(t)-2)):
        grams.add(t[i:i+3])
    bits = np.zeros(nBits, dtype=np.float32)
    for g in grams:
        idx = (hash(g) % nBits + nBits) % nBits
        bits[idx] = 1.0
    return bits

def build_membership_and_amine_bits(enum_path: str, out_dir: str):
    """
    Build:
      - membership: ProductName -> amine_key (which is Amine_Name)
      - A_bits:     per-amine fingerprint (shared across all products that use that amine)
    """
    enum_df = pd.read_excel(enum_path) if enum_path.endswith(".xlsx") else pd.read_csv(enum_path)
    for c in ["ProductName","Amine_Name","Amine_SMILES"]:
        if c not in enum_df.columns:
            enum_df[c] = np.nan

    # Map each ProductName to its Amine_Name (preserve ProductName multiplicity but amine is shared)
    rows = []
    for _, r in enum_df.iterrows():
        pnames = split_products(r["ProductName"])
        pname  = pnames if pnames else [str(r["ProductName"]).strip()]
        am_nm  = str(r["Amine_Name"] or "").strip()
        if not am_nm:  # skip if we truly don't know the amine name
            continue
        for pn in pname:
            if pn:
                rows.append((pn, am_nm))
    membership = pd.DataFrame(rows, columns=["ProductName","amine_key"]).drop_duplicates().reset_index(drop=True)

    # Unique amines → build one feature vector per amine_key
    uniq_amines = sorted(membership["amine_key"].dropna().astype(str).unique().tolist())

    # For SMILES lookup, make a map Amine_Name -> Amine_SMILES (first non-empty wins)
    name_to_smiles = {}
    for _, r in enum_df.iterrows():
        nm = str(r.get("Amine_Name","") or "").strip()
        sm = str(r.get("Amine_SMILES","") or "").strip()
        if nm and sm and (nm not in name_to_smiles):
            name_to_smiles[nm] = sm

    A_bits = []
    nA_smi = nA_hash = 0
    for nm in tqdm(uniq_amines, desc="Amine features (unique amines)", unit="amine"):
        smi = name_to_smiles.get(nm, "")
        arr = morgan_fp(smi, nBits=1024, radius=2) if (HAVE_RDKIT and smi) else None
        if isinstance(arr, np.ndarray) and arr.shape == (1024,):
            nA_smi += 1
        else:
            arr = name_hash_fp(nm if nm else "UNK_AMINE", nBits=1024)
            nA_hash += 1
        A_bits.append(np.asarray(arr, dtype=np.float32))
    A_bits = np.stack(A_bits, axis=0).astype(np.float32)

    # Assign feat_idx per amine
    amine_df = pd.DataFrame({"amine_key": uniq_amines})
    amine_df["feat_idx"] = np.arange(len(amine_df), dtype=int)
    key2idx = dict(zip(amine_df["amine_key"], amine_df["feat_idx"]))
    membership["feat_idx"] = membership["amine_key"].map(key2idx)

    # Save artifacts
    membership.to_csv(pjoin(out_dir, "membership_product_to_amine.csv"), index=False)
    amine_df.to_csv(pjoin(out_dir, "amine_catalog.csv"), index=False)
    np.savez_compressed(pjoin(out_dir, "amine_features.npz"),
                        A=A_bits, amine_keys=amine_df["amine_key"].values, feat_idx=amine_df["feat_idx"].values)

    print(f"[membership] ProductName unique={membership['ProductName'].nunique()} rows={len(membership)}")
    print(f"[amines]    unique amines={len(uniq_amines)}")
    print(f"[amine bits] SMILES {nA_smi} | name-hash {nA_hash} -> {A_bits.shape}")
    return membership, amine_df, A_bits


# =================== Labels & pairs ===================
def derive_label_vec(df: pd.DataFrame, thr: float) -> pd.Series:
    if "label" in df.columns and df["label"].notna().any():
        return df["label"].astype(float).astype(int)
    if "Intensity" in df.columns:
        return (pd.to_numeric(df["Intensity"], errors="coerce").fillna(-1) > float(thr)).astype(int)
    raise ValueError("Need 'label' or 'Intensity' in heatmap.")

def make_labeled_pairs(heat_long: pd.DataFrame, membership: pd.DataFrame,
                       enzyme_vecs: Dict[str, np.ndarray], intensity_thr: float) -> pd.DataFrame:
    """
    Join heatmap with ProductName→amine_key mapping.
    Deduplicate replicates by (Enzyme, ProductName, amine_feat) with OR over labels.
    """
    hl = heat_long.copy()
    hl["ProductName"] = hl["ProductName"].astype(str)

    df = hl.merge(membership[["ProductName","amine_key","feat_idx"]], on="ProductName", how="left")
    df["label_bin"] = derive_label_vec(df, intensity_thr)
    df["has_emb"]   = df["Enzyme"].astype(str).isin(enzyme_vecs.keys())
    df["has_feat"]  = df["feat_idx"].notna()
    df = df[df["has_emb"] & df["has_feat"]].copy()

    # collapse replicates: if any replicate active -> active
    df = (df.groupby(["Enzyme","ProductName","amine_key","feat_idx"], as_index=False)
            .agg(label_bin=("label_bin","max")))
    return df


# =================== Core/bag = class one-hot + positional one-hot ===================
def parse_name_to_fields(product_name: str):
    return str(product_name or "").split("_")

def extract_positional_tokens(head: str):
    """
    Return sorted tuple among {3a,6a,7a,12a,12k,24a}. Extend if needed.
    Handles forms like '3a,7a,12k' or compact '3a12k'.
    """
    if not head: return tuple()
    head_norm = head.replace("-", ",").replace(" ", ",")
    toks = set()
    for part in head_norm.split(","):
        part = part.strip()
        if not part: continue
        found = re.findall(r"(?:3a|6a|7a|12a|12k|24a)", part, flags=re.I)
        for f in found:
            toks.add(f.lower())
    return tuple(sorted(toks))

def extract_hydroxylation_class(head: str, pos_tokens: Tuple[str,...]):
    h = (head or "").lower()
    if h.startswith("mono"): return "mono"
    if h.startswith("di"):   return "di"
    if h.startswith("tri"):  return "tri"
    n = len(pos_tokens)
    if n <= 1:  return "mono"
    if n == 2:  return "di"
    return "tri"

def build_core_combo_matrices(product_names: pd.Series):
    N = len(product_names)
    heads = [parse_name_to_fields(p)[0] if parse_name_to_fields(p) else "" for p in product_names.tolist()]
    pos_list = [extract_positional_tokens(h) for h in heads]
    cls_list = [extract_hydroxylation_class(h, pos_list[i]) for i, h in enumerate(heads)]

    # class one-hot
    CLS = ["mono","di","tri"]
    C_cls = np.zeros((N, len(CLS)), dtype=np.float32)
    for i, c in enumerate(cls_list):
        C_cls[i, CLS.index(c)] = 1.0

    # positional one-hot over unique positional sets
    pos_keys = []
    for toks in pos_list:
        pos_keys.append("pos:" + "_".join(toks) if toks else "pos:none")
    uniq_pos = sorted(set(pos_keys))
    pos2idx = {k:i for i,k in enumerate(uniq_pos)}
    C_pos = np.zeros((N, len(uniq_pos)), dtype=np.float32)
    for i, k in enumerate(pos_keys):
        C_pos[i, pos2idx[k]] = 1.0

    C = np.concatenate([C_cls, C_pos], axis=1)
    meta = {"mode": "combo_cls_pos", "class_labels": CLS, "positional_keys": uniq_pos}
    return C, meta


# =================== Build features ===================
def build_feature_matrix(full_df: pd.DataFrame,
                         enzyme_vecs: Dict[str, np.ndarray],
                         A_bits: np.ndarray,
                         idx_for_amine_key: Dict[str,int]):
    n = len(full_df)
    d_prot = next(iter(enzyme_vecs.values())).shape[0]
    d_A    = A_bits.shape[1]

    E = np.zeros((n, d_prot), dtype=np.float32)
    Af = np.zeros((n, d_A),    dtype=np.float32)
    y  = np.zeros(n, dtype=np.int64)
    groups = np.empty(n, dtype=object)

    it = full_df[["Enzyme","amine_key","label_bin"]].to_numpy()
    keep_idx = []
    for i, (e, akey, lbl) in enumerate(tqdm(it, total=len(it), desc="Assembling base features", unit="row")):
        e = str(e); akey = str(akey)
        if e not in enzyme_vecs: continue
        if akey not in idx_for_amine_key: continue
        ir = idx_for_amine_key[akey]
        E[i] = enzyme_vecs[e].astype(np.float32)
        Af[i] = A_bits[ir]
        y[i] = int(lbl)
        groups[i] = e
        keep_idx.append(i)

    # compact
    keep_idx = np.array(keep_idx, dtype=int)
    E, Af, y, groups = E[keep_idx], Af[keep_idx], y[keep_idx], groups[keep_idx]
    df_kept = full_df.iloc[keep_idx].reset_index(drop=True)

    # core combo from ProductName
    C, core_meta = build_core_combo_matrices(df_kept["ProductName"])

    X = np.concatenate([E, Af, C], axis=1)
    d_core = C.shape[1]
    return X, y, groups, d_prot, d_A, d_core, core_meta


# =================== Training utils ===================
def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    classes = np.unique(y)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(wi) for c, wi in zip(classes, w)}

def calibrate_or_fallback(estimator, X, y, cv=3):
    try:
        cal = CalibratedClassifierCV(estimator=estimator, method="isotonic", cv=cv)
        cal.fit(X, y); return cal
    except Exception:
        cal = CalibratedClassifierCV(estimator=estimator, method="sigmoid", cv=cv)
        cal.fit(X, y); return cal

def split_by_enzyme(groups_vec, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=42):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    enzymes = np.unique(groups_vec)
    rng.shuffle(enzymes)
    n = len(enzymes)
    n_train = int(round(train_frac * n))
    n_val   = int(round(val_frac   * n))
    enz_train = set(enzymes[:n_train])
    enz_val   = set(enzymes[n_train:n_train+n_val])
    enz_test  = set(enzymes[n_train+n_val:])
    assert enz_train.isdisjoint(enz_val) and enz_train.isdisjoint(enz_test) and enz_val.isdisjoint(enz_test)
    train_mask = np.array([g in enz_train for g in groups_vec], dtype=bool)
    val_mask   = np.array([g in enz_val   for g in groups_vec], dtype=bool)
    test_mask  = np.array([g in enz_test  for g in groups_vec], dtype=bool)
    return train_mask, val_mask, test_mask, (enz_train, enz_val, enz_test)

def repeated_group_cv(name, model_ctor, X, y, groups, n_splits, n_repeats, seed,
                      metrics_csv_path, preds_csv_path):
    rng = np.random.default_rng(seed)
    ap_all, roc_all, f1_all, acc_all = [], [], [], []

    for rep in range(n_repeats):
        print(f"\n--- {name} Repeat {rep+1}/{n_repeats} ---")
        shuffled_groups = sk_shuffle(groups, random_state=int(rng.integers(0, 1_000_000_000)))
        gkf = GroupKFold(n_splits=n_splits)
        for fold, (tr, va) in enumerate(gkf.split(X, y, shuffled_groups), 1):
            clf = model_ctor()
            t0 = time.time()
            clf.fit(X[tr], y[tr])
            p = clf.predict_proba(X[va])[:, 1]
            yhat = (p >= 0.5).astype(int)

            ap  = average_precision_score(y[va], p)
            roc = roc_auc_score(y[va], p)
            f1  = f1_score(y[va], yhat)
            acc = accuracy_score(y[va], yhat)
            ap_all.append(ap); roc_all.append(roc); f1_all.append(f1); acc_all.append(acc)
            print(f"    fold {fold}/{n_splits} {time.time()-t0:.1f}s  AP={ap:.3f}")

            # fold metrics
            _append_row(metrics_csv_path,
                        ["model","repeat","fold","ap","roc","f1","acc","n_val"],
                        {"model": name, "repeat": rep, "fold": fold,
                         "ap": ap, "roc": roc, "f1": f1, "acc": acc, "n_val": int(len(va))})

            # per-example predictions
            write_header = not os.path.exists(preds_csv_path)
            with open(preds_csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["model","repeat","fold","y_true","y_score"])
                if write_header:
                    w.writeheader()
                    write_header = False
                for yi, pi in zip(y[va], p):
                    w.writerow({"model": name, "repeat": rep, "fold": fold,
                                "y_true": int(yi), "y_score": float(pi)})

    return {
        "PR_AUC": (float(np.mean(ap_all)),  float(np.std(ap_all))),
        "ROC_AUC":(float(np.mean(roc_all)), float(np.std(roc_all))),
        "F1":     (float(np.mean(f1_all)),  float(np.std(f1_all))),
        "ACC":    (float(np.mean(acc_all)), float(np.std(acc_all))),
    }


# =================== MAIN ===================
def main():
    args = parse_args()
    set_seed(args.seed)

    from datetime import datetime
    run_tag = args.run_tag if args.run_tag else datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = pjoin(os.path.abspath(args.out_dir), run_tag)
    os.makedirs(out_dir, exist_ok=True)

    cfg = TrainConfig(
        data_dir=os.path.abspath(args.data_dir),
        out_dir=out_dir,
        n_splits=args.splits,
        n_repeats=args.repeats,
        seed=args.seed,
        intensity_threshold=args.intensity_threshold,
        use_rf=(not args.no_rf),
        use_et=args.et,
        calib_cv=args.calib_cv,
        save_cache=args.save_cache,
        load_cache=args.load_cache,
    )

    # paths
    H5_PATH   = pjoin(cfg.data_dir, "Seqs_list_total.h5")
    HEAT_CSV  = pjoin(cfg.data_dir, "ipsita_heatmap_long.csv")
    ENUM_XLSX = pjoin(cfg.data_dir, "swap_enumeration_FINAL.xlsx")
    CACHE_NPZ = pjoin(cfg.out_dir, "Xy_groups_cache.npz")

    # output logs
    METRICS_CSV = pjoin(cfg.out_dir, "cv_fold_metrics.csv")
    PREDS_CSV   = pjoin(cfg.out_dir, "cv_predictions.csv")
    for pth in [METRICS_CSV, PREDS_CSV]:
        if os.path.exists(pth): os.remove(pth)

    # enzyme embeddings
    print("== Building/loading enzyme embeddings ==")
    enzyme_vecs = load_or_build_enzyme_embeddings(H5_PATH, HEAT_CSV, cfg.out_dir)
    d_prot = next(iter(enzyme_vecs.values())).shape[0]
    print(f"  Enzymes: {len(enzyme_vecs)}  dim={d_prot}")

    # ProductName→Amine_Name + unique amine bits
    print("== Membership (ProductName→Amine_Name) + amine bits (no BA fp) ==")
    membership, amine_df, A_bits = build_membership_and_amine_bits(ENUM_XLSX, cfg.out_dir)
    key2idx = dict(zip(amine_df["amine_key"].astype(str), amine_df["feat_idx"].astype(int)))
    d_A = A_bits.shape[1]

    # labeled pairs
    print("== Labeled pairs from heatmap (dedup by enzyme/product/amine) ==")
    heat_long = pd.read_csv(HEAT_CSV)
    pairs_df = make_labeled_pairs(heat_long, membership, enzyme_vecs, cfg.intensity_threshold)
    print(f"  rows={len(pairs_df)} enzymes={pairs_df['Enzyme'].nunique()} unique_products={pairs_df['ProductName'].nunique()} unique_amines={pairs_df['amine_key'].nunique()}")

    # features
    if cfg.load_cache and os.path.exists(CACHE_NPZ):
        print("== Loading cached features ==")
        cache = np.load(CACHE_NPZ, allow_pickle=True)
        X, y, groups = cache["X"], cache["y"], cache["groups"]
        d_core = int(cache["d_core"])
        core_meta = json.load(open(pjoin(cfg.out_dir, "core_meta.json")))
    else:
        print("== Building X = [enzyme || amine || (mono/di/tri || positional)] ==")
        X, y, groups, _, _, d_core, core_meta = build_feature_matrix(
            pairs_df, enzyme_vecs, A_bits, key2idx
        )
        if cfg.save_cache:
            np.savez_compressed(CACHE_NPZ, X=X, y=y, groups=groups, d_core=d_core)
            with open(pjoin(cfg.out_dir, "core_meta.json"), "w") as f:
                json.dump(core_meta, f, indent=2)

    print(f"[Features] X={X.shape} (d_prot={d_prot}, d_A={d_A}, d_core={d_core})  y={y.shape}  groups={len(np.unique(groups))}")

    # ================= Repeated GroupKFold: compare models =================
    results = {}

    # Precompute scaled features for LR/MLP
    scaler = StandardScaler(with_mean=True, with_std=True).fit(X)
    X_scaled = scaler.transform(X).astype(np.float32)
    cw = compute_class_weights(y)

    # Model factories
    def mk_lr():
        return LogisticRegression(max_iter=5000, class_weight=cw)
    def mk_mlp():
        return MLPClassifier(hidden_layer_sizes=(128, 64),
                             activation="relu", alpha=1e-4, learning_rate_init=1e-3,
                             max_iter=100, early_stopping=True, n_iter_no_change=10,
                             random_state=cfg.seed)
    def mk_rf():
        return RandomForestClassifier(n_estimators=350, class_weight="balanced", n_jobs=-1, random_state=cfg.seed)
    def mk_et():
        return ExtraTreesClassifier(n_estimators=600, class_weight="balanced", n_jobs=-1, random_state=cfg.seed)
    def mk_xgb():
        neg = float((y == 0).sum()); pos = float((y == 1).sum()); spw = max(neg/max(pos,1.0), 1.0)
        return XGBClassifier(
            n_estimators=500, max_depth=7, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.7, reg_lambda=1.0,
            objective="binary:logistic", tree_method="hist", max_bin=256,
            eval_metric="aucpr", random_state=cfg.seed, n_jobs=-1,
            scale_pos_weight=spw
        )

    print("\n== Repeated GroupKFold (enzyme) with prediction logs ==")
    results["logreg"] = repeated_group_cv("logreg", mk_lr,  X_scaled, y, groups,
                                          n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, seed=cfg.seed,
                                          metrics_csv_path=METRICS_CSV, preds_csv_path=PREDS_CSV)
    results["mlp"]    = repeated_group_cv("mlp",    mk_mlp, X_scaled, y, groups,
                                          n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, seed=cfg.seed,
                                          metrics_csv_path=METRICS_CSV, preds_csv_path=PREDS_CSV)
    if cfg.use_rf:
        results["rf"] = repeated_group_cv("rf", mk_rf, X, y, groups,
                                          n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, seed=cfg.seed,
                                          metrics_csv_path=METRICS_CSV, preds_csv_path=PREDS_CSV)
    if cfg.use_et:
        results["et"] = repeated_group_cv("et", mk_et, X, y, groups,
                                          n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, seed=cfg.seed,
                                          metrics_csv_path=METRICS_CSV, preds_csv_path=PREDS_CSV)
    results["xgb"]    = repeated_group_cv("xgb",    mk_xgb, X, y, groups,
                                          n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, seed=cfg.seed,
                                          metrics_csv_path=METRICS_CSV, preds_csv_path=PREDS_CSV)

    # Winner by mean PR-AUC
    winner = max(results.keys(), key=lambda k: results[k]["PR_AUC"][0])

    print("\n=== Repeated GroupKFold Summary (mean±std across all folds×repeats) ===")
    for k, m in results.items():
        ap, aps = m["PR_AUC"]; roc, rocs = m["ROC_AUC"]; f1v, f1s = m["F1"]; accm, accs = m["ACC"]
        print(f"{k:>6} | PR-AUC {ap:.3f}±{aps:.3f} | ROC {roc:.3f}±{rocs:.3f} | F1 {f1v:.3f}±{f1s:.3f} | ACC {accm:.3f}±{accs:.3f}")
    print(f"[Winner] {winner}")

    # Final fit on ALL data + calibrate
    print("\n== Fitting final winner on ALL data and calibrating ==")
    if winner in ("logreg","mlp"):
        final_base = {"logreg": mk_lr, "mlp": mk_mlp}[winner]()
        final_base.fit(X_scaled, y)
        final_cal = calibrate_or_fallback(final_base, X_scaled, y, cv=cfg.calib_cv)
        need_scaler = True
    else:
        final_base = {"rf": mk_rf, "et": mk_et, "xgb": mk_xgb}[winner]()
        final_base.fit(X, y)
        final_cal = calibrate_or_fallback(final_base, X, y, cv=cfg.calib_cv)
        need_scaler = False

    # Save
    import joblib
    joblib.dump(final_cal, pjoin(cfg.out_dir, f"{winner}_calibrated.joblib"))
    if need_scaler:
        joblib.dump(scaler, pjoin(cfg.out_dir, "scaler.joblib"))
    with open(pjoin(cfg.out_dir, "repeated_cv_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    meta = {
        "config": asdict(cfg),
        "dims": {"enzyme_emb": int(d_prot), "amine_bits": int(d_A), "core": int(d_core), "X_dim": int(X.shape[1])},
        "core_meta": core_meta,
        "winner": winner
    }
    with open(pjoin(cfg.out_dir, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Optional: fixed enzyme hold-out
    if args.holdout:
        try:
            trf, vf, tf = map(float, args.holdout.split(","))
        except Exception:
            raise SystemExit("Use --holdout like '0.7,0.15,0.15'")
        ho_metrics = []
        print(f"\n== Fixed enzyme hold-out: train={trf}, val={vf}, test={tf} (repeats={args.holdout_repeats}) ==")
        for rep in range(args.holdout_repeats):
            print(f"\n-- Holdout repeat {rep+1}/{args.holdout_repeats} --")
            train_mask, val_mask, test_mask, (enz_tr, enz_val, enz_te) = split_by_enzyme(
                groups, trf, vf, tf, seed=cfg.seed + rep
            )
            Xtr, ytr = X[train_mask], y[train_mask]
            Xva, yva = X[val_mask],   y[val_mask]
            Xte, yte = X[test_mask],  y[test_mask]

            neg = float((ytr == 0).sum()); pos = float((ytr == 1).sum()); spw = max(neg/max(pos,1.0), 1.0)
            xgb_ho = XGBClassifier(
                n_estimators=800, max_depth=7, learning_rate=0.05,
                subsample=0.9, colsample_bytree=0.7, reg_lambda=1.0,
                objective="binary:logistic", tree_method="hist", max_bin=256,
                eval_metric="aucpr", random_state=cfg.seed + rep, n_jobs=-1,
                scale_pos_weight=spw
            )
            # train+val, then test once
            xgb_ho.fit(np.vstack([Xtr, Xva]), np.hstack([ytr, yva]))
            xgb_cal = calibrate_or_fallback(xgb_ho, np.vstack([Xtr, Xva]), np.hstack([ytr, yva]), cv=cfg.calib_cv)

            p = xgb_cal.predict_proba(Xte)[:, 1]
            yhat = (p >= 0.5).astype(int)
            ap  = average_precision_score(yte, p)
            roc = roc_auc_score(yte, p)
            f1  = f1_score(yte, yhat)
            acc = accuracy_score(yte, yhat)
            ho_metrics.append({"pr_auc": float(ap), "roc_auc": float(roc), "f1": float(f1), "acc": float(acc),
                               "enz_counts": {"train": len(enz_tr), "val": len(enz_val), "test": len(enz_te)}})
            print(f"  TEST metrics: PR-AUC {ap:.3f} | ROC {roc:.3f} | F1 {f1:.3f} | ACC {acc:.3f}")

        # save holdout
        agg = {
            "repeats": args.holdout_repeats,
            "mean": {k: float(np.mean([m[k] for m in ho_metrics])) for k in ["pr_auc","roc_auc","f1","acc"]},
            "std":  {k: float(np.std([m[k] for m in ho_metrics]))  for k in ["pr_auc","roc_auc","f1","acc"]},
            "details": ho_metrics
        }
        with open(pjoin(cfg.out_dir, "holdout_results.json"), "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\n[Saved] {pjoin(cfg.out_dir, 'holdout_results.json')}")

    # Inference helper (uses amine catalog keyed by Amine_Name)
    infer_py = f"""\
import joblib, numpy as np, pandas as pd, json, re

OUT_DIR = r"{cfg.out_dir}"
model = joblib.load(f"{{OUT_DIR}}/{winner}_calibrated.joblib")
try:
    scaler = joblib.load(f"{{OUT_DIR}}/scaler.joblib")
except Exception:
    scaler = None

embs = np.load(f"{{OUT_DIR}}/enzyme_embeddings.npy", allow_pickle=True).item()
mem  = pd.read_csv(f"{{OUT_DIR}}/membership_product_to_amine.csv")  # ProductName -> amine_key -> feat_idx
amine_cat = pd.read_csv(f"{{OUT_DIR}}/amine_catalog.csv")
A_npz = np.load(f"{{OUT_DIR}}/amine_features.npz", allow_pickle=True)
A, amine_keys, feat_idx = A_npz["A"], A_npz["amine_keys"], A_npz["feat_idx"]
key2idx = {{str(k): int(fi) for k, fi in zip(amine_keys, feat_idx)}}

core_meta = json.load(open(f"{{OUT_DIR}}/core_meta.json"))
CLS = core_meta["class_labels"]
POS_KEYS = core_meta["positional_keys"]

def parse_name_to_fields(product_name: str):
    return str(product_name or "").split("_")

def extract_positional_tokens(head: str):
    if not head: return tuple()
    head_norm = head.replace("-", ",").replace(" ", ",")
    toks = set()
    for part in head_norm.split(","):
        part = part.strip()
        if not part: continue
        found = re.findall(r"(?:3a|6a|7a|12a|12k|24a)", part, flags=re.I)
        for f in found:
            toks.add(f.lower())
    return tuple(sorted(toks))

def extract_hydroxylation_class(head: str, pos_tokens):
    h = (head or "").lower()
    if h.startswith("mono"): return "mono"
    if h.startswith("di"):   return "di"
    if h.startswith("tri"):  return "tri"
    n = len(pos_tokens)
    if n <= 1: return "mono"
    if n == 2: return "di"
    return "tri"

def build_core_vec(product_name: str):
    chunks = parse_name_to_fields(product_name)
    head = chunks[0] if chunks else ""
    pos_toks = extract_positional_tokens(head)
    klass = extract_hydroxylation_class(head, pos_toks)
    cls_v = np.zeros((1, len(CLS)), dtype=np.float32); cls_v[0, CLS.index(klass)] = 1.0
    pos_key = "pos:" + "_".join(pos_toks) if pos_toks else "pos:none"
    pos_v = np.zeros((1, len(POS_KEYS)), dtype=np.float32)
    if pos_key in POS_KEYS:
        pos_v[0, POS_KEYS.index(pos_key)] = 1.0
    return np.concatenate([cls_v, pos_v], axis=1)

def score(enzyme: str, product_name: str):
    if enzyme not in embs: raise KeyError(f"unknown enzyme: {{enzyme}}")
    row = mem.loc[mem["ProductName"]==product_name]
    if row.empty: raise KeyError(f"unknown ProductName: {{product_name}}")
    akey = str(row.iloc[0]["amine_key"])
    if akey not in key2idx: raise KeyError(f"unknown amine_key: {{akey}}")
    ir = key2idx[akey]
    E = embs[enzyme].astype(np.float32)[None, :]
    Arow = A[ir][None, :]
    C = build_core_vec(product_name)
    X = np.concatenate([E, Arow, C], axis=1)
    if scaler is not None: X = scaler.transform(X)
    return float(model.predict_proba(X)[0,1])
"""
    with open(pjoin(cfg.out_dir, "inference_example.py"), "w") as f:
        f.write(infer_py)

    # convenience: save enzyme embeddings used
    np.save(pjoin(cfg.out_dir, "enzyme_embeddings.npy"), enzyme_vecs)

    print(f"\n[Saved] {pjoin(cfg.out_dir, 'repeated_cv_results.json')}")
    print(f"[Saved] {pjoin(cfg.out_dir, 'cv_fold_metrics.csv')}")
    print(f"[Saved] {pjoin(cfg.out_dir, 'cv_predictions.csv')}")
    print(f"[Saved] {pjoin(cfg.out_dir, 'training_meta.json')}")
    print(f"[Saved] {pjoin(cfg.out_dir, f'{winner}_calibrated.joblib')}")
    if os.path.exists(pjoin(cfg.out_dir, "scaler.joblib")):
        print(f"[Saved] {pjoin(cfg.out_dir, 'scaler.joblib')}")
    print(f"[Saved] {pjoin(cfg.out_dir, 'inference_example.py')}")
    print("Done.")


if __name__ == "__main__":
    main()
