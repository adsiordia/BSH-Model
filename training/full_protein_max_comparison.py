#!/usr/bin/env python3
"""
Full-protein max pooling comparison.

Adds `full_protein_max` strategy (all residues, max-pooled) alongside:
  - full_protein_mean  (all residues, mean-pooled)
  - noncons_mean       (non-conserved residues, mean-pooled)
  - noncons_max        (non-conserved residues, max-pooled)

Runs 10-seed enzyme hold-out evaluation with XGBoost, RF, MLP.
Saves results CSV + comparison plots.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import pickle
import warnings
warnings.filterwarnings("ignore")

from Bio import SeqIO
from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    log_loss
)

import xgboost as xgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── paths ──────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUT        = OUTPUT_DIR / "full_protein_max_comparison"
OUT.mkdir(exist_ok=True, parents=True)

H5_PER_RESIDUE = DATA_DIR / "Seqs_list_total_per_residue.h5"
H5_FULL        = DATA_DIR / "Seqs_list_total.h5"
ALIGNMENT_PATH = OUTPUT_DIR / "bsh_aligned.fasta"
CONS_PATH      = OUTPUT_DIR / "conservation_scores.csv"
ACTIVITY_PATH  = OUTPUT_DIR / "enzyme_amine_activity.csv"
SMILES_PATH    = DATA_DIR / "bsh_reactants_SMILES_corrected.xlsx"

FP_BITS        = 1024
BEST_THRESH    = 0.5
N_SPLITS       = 10
SEEDS          = [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066]

# ── 1. Load per-residue embeddings ─────────────────────────────────
print("Loading per-residue embeddings …")
per_residue_embeddings = {}
with h5py.File(H5_PER_RESIDUE, "r") as f:
    for key in f.keys():
        data = f[key][:]
        uid = key.split("_")[-1] if "_" in key else key
        per_residue_embeddings[uid] = data
print(f"  {len(per_residue_embeddings)} enzymes, e.g. shape={next(iter(per_residue_embeddings.values())).shape}")

# ── 2. Load pre-pooled (mean) full-protein embeddings ──────────────
print("Loading pre-pooled full-protein embeddings …")
full_mean_embeddings = {}
with h5py.File(H5_FULL, "r") as f:
    for key in f.keys():
        uid = key.split("_")[-1] if "_" in key else key
        full_mean_embeddings[uid] = np.array(f[key]).astype(np.float32)
print(f"  {len(full_mean_embeddings)} enzymes")

# ── 3. Conservation scores + alignment mapping ────────────────────
print("Loading conservation scores …")
df_cons = pd.read_csv(CONS_PATH)
df_core = df_cons[~df_cons["is_high_gap"]].copy()
print(f"  {len(df_core)} core alignment positions (low-gap)")

print("Building alignment-to-sequence mapping …")
alignment_to_seq = {}
for record in SeqIO.parse(ALIGNMENT_PATH, "fasta"):
    parts = record.id.split("_")
    uid = parts[-1] if len(parts) > 1 else record.id
    seq = str(record.seq)
    mapping = {}
    seq_pos = 0
    for aln_pos, char in enumerate(seq):
        if char != "-":
            mapping[aln_pos] = seq_pos
            seq_pos += 1
    alignment_to_seq[uid] = mapping
overlap = set(alignment_to_seq.keys()) & set(per_residue_embeddings.keys())
print(f"  {len(alignment_to_seq)} aligned, {len(overlap)} with embeddings")

# ── 4. Activity labels ────────────────────────────────────────────
print("Loading activity labels …")
df_activity = pd.read_csv(ACTIVITY_PATH)
controls = ["CTRL1","CTRL2","CTRL3","CTRL4","CTRL5","CTRL6","CTRL7"]
canonical = ["taurine", "glycine"]
df_activity = df_activity[~df_activity["Enzyme"].isin(controls)]
df_activity = df_activity[~df_activity["Amine"].isin(canonical)]

df_agg = df_activity.groupby(["Enzyme", "Amine"]).agg(
    active=("active_approach2", "any"),
    n_products=("ProductName", "count"),
    n_active_products=("active_approach2", "sum"),
).reset_index()
print(f"  {df_agg.shape[0]} enzyme-amine pairs, {df_agg['active'].sum()} active ({100*df_agg['active'].mean():.1f}%)")

# ── 5. Amine fingerprints ─────────────────────────────────────────
print("Building amine fingerprints …")
name_map = {
    "2,3-Diaminopropinoic Acid": "2,3_diaminopropionic acid",
    "2-aminophenol": "2_aminophenol",
    "3-methoxytyramine HCl": "3_methoxytyramine",
    "4-aminophenol": "4_aminophenol",
    "L-Alanine": "alanine", "L-Arginine": "arginine",
    "Asparagine": "asparagine", "Cadaverine": "cadaverine",
    "L-Citrulline": "citrulline", "L-Cysteine": "cysteine",
    "Dopamine HCl": "dopamine",
    "gamma-Aminobutyric acid >99%": "gaba",
    "L-Glutamine": "glutamine", "Glycyl-L-Valine": "glyglycine",
    "L-Histidine": "histidine", "L-Lysine": "lysine",
    "L-Methionine": "methionine",
    "L-Ornithine monohydrochloride": "ornithine",
    "L-Phenylalanine": "phenylalanine", "DL-Proline": "proline",
    "Putrescine": "putrescine", "L-Serine": "serine",
    "L-Threonine": "threonine", "Tryptamine": "tryptamine",
}

df_smiles = pd.read_excel(SMILES_PATH)
amine_fingerprints = {}
for _, row in df_smiles.iterrows():
    name = row["Compound_Name"]
    smiles = row["SMILES"]
    norm = name_map.get(name, name.lower().replace(" ", "_").replace("-", "_"))
    if pd.isna(smiles):
        continue
    mol = Chem.MolFromSmiles(smiles.split(".")[0])
    if mol is not None:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=FP_BITS)
        amine_fingerprints[norm] = np.array(fp, dtype=np.float32)
for a in df_agg["Amine"].unique():
    if a not in amine_fingerprints:
        amine_fingerprints[a] = np.zeros(FP_BITS, dtype=np.float32)
print(f"  {len(amine_fingerprints)} amines")

# ── 6. Pooling helpers ─────────────────────────────────────────────

def get_nonconserved_embedding(enzyme_id, threshold, pooling="mean"):
    if enzyme_id not in per_residue_embeddings or enzyme_id not in alignment_to_seq:
        return None
    embed = per_residue_embeddings[enzyme_id]
    aln_map = alignment_to_seq[enzyme_id]
    variable_pos = df_core[df_core["conservation_score"] < threshold]["alignment_position"].values
    seq_positions = [aln_map[p] for p in variable_pos if p in aln_map and aln_map[p] < len(embed)]
    if len(seq_positions) == 0:
        return None
    selected = embed[seq_positions]
    if pooling == "mean":
        return selected.mean(axis=0).astype(np.float32)
    elif pooling == "max":
        return selected.max(axis=0).astype(np.float32)
    return selected.mean(axis=0).astype(np.float32)


def get_full_protein_embedding(enzyme_id, pooling="mean"):
    """Pool ALL residues of the full protein (from per-residue embeddings)."""
    if enzyme_id not in per_residue_embeddings:
        return None
    embed = per_residue_embeddings[enzyme_id]  # (seq_len, 1024)
    if pooling == "mean":
        return embed.mean(axis=0).astype(np.float32)
    elif pooling == "max":
        return embed.max(axis=0).astype(np.float32)
    return embed.mean(axis=0).astype(np.float32)

# ── 7. Build strategy feature matrices ─────────────────────────────
print("\nBuilding strategy feature matrices …")

strategies = {}

for strat_name, pool_func, pool_arg in [
    ("full_protein_mean", get_full_protein_embedding, "mean"),
    ("full_protein_max",  get_full_protein_embedding, "max"),
    ("noncons_mean",      lambda e, p: get_nonconserved_embedding(e, BEST_THRESH, p), "mean"),
    ("noncons_max",       lambda e, p: get_nonconserved_embedding(e, BEST_THRESH, p), "max"),
]:
    X_list, y_list, enz_list, ami_list = [], [], [], []
    for _, row in df_agg.iterrows():
        enzyme, amine = row["Enzyme"], row["Amine"]
        emb = pool_func(enzyme, pool_arg)
        if emb is None or amine not in amine_fingerprints:
            continue
        X_list.append(np.concatenate([emb, amine_fingerprints[amine]]))
        y_list.append(int(row["active"]))
        enz_list.append(enzyme)
        ami_list.append(amine)
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    strategies[strat_name] = (X, y, enz_list, ami_list)
    n_active = int(y.sum())
    print(f"  {strat_name:25s}: {X.shape}, active={n_active} ({100*n_active/len(y):.1f}%)")

# ── 8. Enzyme hold-out evaluation ─────────────────────────────────

def enzyme_holdout_split(X, y, enzymes, amines, seed, test_size=0.2, val_size=0.2):
    enz_arr = np.array(enzymes)
    unique_enz = np.unique(enz_arr)
    profiles = np.array([y[enz_arr == e].mean() for e in unique_enz])
    bins = pd.cut(profiles, bins=5, labels=False)
    tv_enz, te_enz = train_test_split(unique_enz, test_size=test_size, random_state=seed, stratify=bins)
    tv_prof = np.array([y[enz_arr == e].mean() for e in tv_enz])
    tv_bins = pd.cut(tv_prof, bins=5, labels=False)
    tr_enz, va_enz = train_test_split(tv_enz, test_size=val_size, random_state=seed, stratify=tv_bins)
    tr_m = np.isin(enz_arr, tr_enz)
    va_m = np.isin(enz_arr, va_enz)
    te_m = np.isin(enz_arr, te_enz)
    return dict(
        X_train=X[tr_m], y_train=y[tr_m],
        X_val=X[va_m],   y_val=y[va_m],
        X_test=X[te_m],  y_test=y[te_m],
    )


def train_evaluate(split, model_type):
    Xtr, ytr = split["X_train"], split["y_train"]
    Xva, yva = split["X_val"],   split["y_val"]
    Xte, yte = split["X_test"],  split["y_test"]
    n_neg = int((ytr == 0).sum())
    n_pos = int((ytr == 1).sum())
    spw = max(n_neg / max(n_pos, 1), 1.0)

    if model_type == "XGBoost":
        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            scale_pos_weight=spw, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, early_stopping_rounds=20,
            eval_metric="logloss", n_jobs=-1, verbosity=0,
        )
        clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    elif model_type == "RF":
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=20, min_samples_leaf=5,
            class_weight={0: 1.0, 1: spw}, random_state=42, n_jobs=-1,
        )
        clf.fit(Xtr, ytr)
    elif model_type == "MLP":
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)
        Xva = scaler.transform(Xva)
        Xte = scaler.transform(Xte)
        clf = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation="relu",
            alpha=0.01, batch_size=64, learning_rate_init=0.001,
            max_iter=500, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=10, random_state=42, verbose=False,
        )
        clf.fit(Xtr, ytr)
    else:
        raise ValueError(model_type)

    y_pred  = clf.predict(Xte)
    y_proba = clf.predict_proba(Xte)[:, 1]
    train_proba = clf.predict_proba(Xtr)[:, 1]

    return dict(
        train_acc=float(accuracy_score(ytr, clf.predict(Xtr))),
        test_acc=float(accuracy_score(yte, y_pred)),
        balanced_acc=float(balanced_accuracy_score(yte, y_pred)),
        precision=float(precision_score(yte, y_pred, zero_division=0)),
        recall=float(recall_score(yte, y_pred, zero_division=0)),
        f1=float(f1_score(yte, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(yte, y_proba)),
        pr_auc=float(average_precision_score(yte, y_proba)),
        test_logloss=float(log_loss(yte, np.clip(y_proba, 1e-6, 1-1e-6))),
        train_logloss=float(log_loss(ytr, np.clip(train_proba, 1e-6, 1-1e-6))),
    )

# ── 9. Run 10-seed evaluation ─────────────────────────────────────
print(f"\nRunning {N_SPLITS}-seed enzyme hold-out evaluation …")
results = []

for strat_name, (X, y, enzymes, amines) in strategies.items():
    for model_type in ["XGBoost", "RF", "MLP"]:
        for i, seed in enumerate(SEEDS):
            split = enzyme_holdout_split(X, y, enzymes, amines, seed)
            metrics = train_evaluate(split, model_type)
            metrics["strategy"] = strat_name
            metrics["model"] = model_type
            metrics["seed"] = seed
            metrics["split_idx"] = i
            results.append(metrics)
            if i == 0:
                print(f"  {strat_name:25s} | {model_type:8s} | seed {seed}: "
                      f"ROC={metrics['roc_auc']:.3f}  PR={metrics['pr_auc']:.3f}  F1={metrics['f1']:.3f}")
        # Print mean after all seeds
        strat_model = [r for r in results if r["strategy"] == strat_name and r["model"] == model_type]
        roc_vals = [r["roc_auc"] for r in strat_model]
        pr_vals  = [r["pr_auc"]  for r in strat_model]
        print(f"  {strat_name:25s} | {model_type:8s} | MEAN: "
              f"ROC={np.mean(roc_vals):.3f}±{np.std(roc_vals):.3f}  "
              f"PR={np.mean(pr_vals):.3f}±{np.std(pr_vals):.3f}")

df_results = pd.DataFrame(results)
df_results.to_csv(OUT / "full_protein_max_comparison_results.csv", index=False)
print(f"\nSaved: {OUT / 'full_protein_max_comparison_results.csv'}")

# ── 10. Summary table ──────────────────────────────────────────────
print("\n" + "=" * 90)
print("SUMMARY: Mean ± Std across 10 seeds (XGBoost)")
print("=" * 90)
print(f"{'Strategy':25s} {'ROC-AUC':>15s} {'PR-AUC':>15s} {'F1':>15s} {'LL Gap':>12s}")
print("-" * 90)

for strat in ["full_protein_mean", "full_protein_max", "noncons_mean", "noncons_max"]:
    mask = (df_results["strategy"] == strat) & (df_results["model"] == "XGBoost")
    sub = df_results[mask]
    roc_m, roc_s = sub["roc_auc"].mean(), sub["roc_auc"].std()
    pr_m,  pr_s  = sub["pr_auc"].mean(),  sub["pr_auc"].std()
    f1_m,  f1_s  = sub["f1"].mean(),      sub["f1"].std()
    ll_gap = (sub["test_logloss"] - sub["train_logloss"]).mean()
    print(f"{strat:25s} {roc_m:.3f} ± {roc_s:.3f}   {pr_m:.3f} ± {pr_s:.3f}   {f1_m:.3f} ± {f1_s:.3f}   +{ll_gap:.3f}")

# ── 11. Plots ──────────────────────────────────────────────────────
strat_order = ["full_protein_mean", "full_protein_max", "noncons_mean", "noncons_max"]
strat_labels = ["Full Protein\n(Mean Pool)", "Full Protein\n(Max Pool)",
                "Non-Conserved\n(Mean Pool)", "Non-Conserved\n(Max Pool)"]
colors = ["#3498db", "#e74c3c", "#2ecc71", "#e67e22"]

# --- Box plots: ROC-AUC and PR-AUC for XGBoost ---
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for ax, metric, title in zip(axes, ["roc_auc", "pr_auc", "f1"], ["ROC-AUC", "PR-AUC", "F1"]):
    data = []
    for strat in strat_order:
        mask = (df_results["strategy"] == strat) & (df_results["model"] == "XGBoost")
        data.append(df_results[mask][metric].values)
    bp = ax.boxplot(data, labels=strat_labels, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    ax.set_title(f"{title} (XGBoost, 10 seeds)", fontsize=13, fontweight="bold")
    ax.set_ylabel(title, fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    # Add mean annotation
    for i, d in enumerate(data):
        ax.annotate(f"{np.mean(d):.3f}", xy=(i + 1, np.mean(d)),
                    ha="center", va="bottom", fontsize=9, fontweight="bold", color=colors[i])

plt.tight_layout()
fig.savefig(OUT / "pooling_comparison_boxplots.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT / 'pooling_comparison_boxplots.png'}")

# --- Bar chart: all 3 models ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, metric, title in zip(axes, ["roc_auc", "pr_auc"], ["ROC-AUC", "PR-AUC"]):
    x = np.arange(len(strat_order))
    width = 0.22
    model_colors = {"XGBoost": "#2ecc71", "RF": "#3498db", "MLP": "#e74c3c"}
    for j, model in enumerate(["XGBoost", "RF", "MLP"]):
        means = []
        stds = []
        for strat in strat_order:
            mask = (df_results["strategy"] == strat) & (df_results["model"] == model)
            means.append(df_results[mask][metric].mean())
            stds.append(df_results[mask][metric].std())
        offset = (j - 1) * width
        ax.bar(x + offset, means, width, yerr=stds, label=model,
               color=model_colors[model], edgecolor="black", linewidth=0.5,
               capsize=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(strat_labels, fontsize=9)
    ax.set_ylabel(title, fontsize=12)
    ax.set_title(f"{title} by Pooling Strategy (10 seeds)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0.4, 1.0)

plt.tight_layout()
fig.savefig(OUT / "pooling_comparison_all_models.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT / 'pooling_comparison_all_models.png'}")

# --- Train-Test gap chart ---
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(strat_order))
width = 0.3
mask_xgb = df_results["model"] == "XGBoost"
train_means = [df_results[mask_xgb & (df_results["strategy"] == s)]["train_acc"].mean() for s in strat_order]
test_means  = [df_results[mask_xgb & (df_results["strategy"] == s)]["test_acc"].mean() for s in strat_order]
train_stds  = [df_results[mask_xgb & (df_results["strategy"] == s)]["train_acc"].std() for s in strat_order]
test_stds   = [df_results[mask_xgb & (df_results["strategy"] == s)]["test_acc"].std() for s in strat_order]

ax.bar(x - width/2, train_means, width, yerr=train_stds, label="Train", color="#3498db", alpha=0.8, capsize=3)
ax.bar(x + width/2, test_means,  width, yerr=test_stds,  label="Test",  color="#e74c3c", alpha=0.8, capsize=3)

for i, (tr, te) in enumerate(zip(train_means, test_means)):
    ax.annotate(f"gap={tr-te:.3f}", xy=(i, max(tr, te) + 0.02), ha="center", fontsize=9, color="gray")

ax.set_xticks(x)
ax.set_xticklabels(strat_labels, fontsize=10)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Train vs Test Accuracy — XGBoost (10 seeds)", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.set_ylim(0.5, 1.05)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(OUT / "pooling_comparison_overfit_gap.png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT / 'pooling_comparison_overfit_gap.png'}")

# ── 12. Save best model ───────────────────────────────────────────
# Retrain best config on the first seed with model saved
best_row = df_results[df_results["model"] == "XGBoost"].sort_values("roc_auc", ascending=False).iloc[0]
best_strat = best_row["strategy"]
best_seed = int(best_row["seed"])
print(f"\nBest single run: {best_strat} / XGBoost / seed={best_seed} "
      f"(ROC={best_row['roc_auc']:.3f}, PR={best_row['pr_auc']:.3f})")

X, y, enzymes, amines = strategies[best_strat]
split = enzyme_holdout_split(X, y, enzymes, amines, best_seed)
n_neg = int((split["y_train"] == 0).sum())
n_pos = int((split["y_train"] == 1).sum())
spw = max(n_neg / max(n_pos, 1), 1.0)

best_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    scale_pos_weight=spw, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, early_stopping_rounds=20,
    eval_metric="logloss", n_jobs=-1, verbosity=0,
)
best_model.fit(split["X_train"], split["y_train"],
               eval_set=[(split["X_val"], split["y_val"])], verbose=False)

model_path = OUT / "best_xgb_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump({
        "model": best_model,
        "strategy": best_strat,
        "seed": best_seed,
        "metrics": best_row.to_dict(),
    }, f)
print(f"Saved: {model_path}")

print("\n✓ Done.")
