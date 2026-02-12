"""
Plot raw train/val log loss curves across boosting rounds
for all 12 combo × label configurations (using seed=42).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from Bio import SeqIO
import xgboost as xgb

# --- Paths ---
DATA_DIR = Path("../data")
OUTPUT_DIR = Path("../outputs")
PRODUCT_DIR = OUTPUT_DIR / "model_outputs" / "product_level_model"

RANDOM_STATE = 42
EXCLUDE_AMINES = {'serotonin', 'tyramine', 'glyglycine', 'cystine', 'unconjugated'}

XGB_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    reg_alpha=1.0, reg_lambda=5.0,
    subsample=0.7, colsample_bytree=0.7,
    min_child_weight=5,
    random_state=42, early_stopping_rounds=30,
    eval_metric='logloss', n_jobs=-1
)

N_PHYSCHEM = 15
PATTERN_INFO = {
    '3a7k':    ('a', 'k', None, 1, 1),
    '3k7a':    ('k', 'a', None, 1, 1),
    '3a12k':   ('a', None, 'k', 1, 1),
    '3k12a':   ('k', None, 'a', 1, 1),
    '3a7a12k': ('a', 'a', 'k',  2, 1),
    'Mono':    (None, None, None, 1, 0),
    'Di':      (None, None, None, 2, 0),
    'Tri':     (None, None, None, 3, 0),
}
BA_FEATURE_NAMES = [
    'C3_aOH', 'C3_keto', 'C3_unspec',
    'C7_aOH', 'C7_keto', 'C7_unspec',
    'C12_aOH', 'C12_keto', 'C12_unspec',
    'n_OH', 'n_keto', 'is_specific',
]
BA_DIM = len(BA_FEATURE_NAMES)

print("Loading data...")

# --- Load embeddings ---
h5_path = DATA_DIR / "Seqs_list_total_per_residue.h5"
per_residue_embeddings = {}
with h5py.File(h5_path, 'r') as f:
    for key in f.keys():
        uniprot_id = key.split('_')[-1]
        per_residue_embeddings[uniprot_id] = f[key][:]

h5_full = DATA_DIR / "Seqs_list_total.h5"
full_embeddings = {}
with h5py.File(h5_full, 'r') as f:
    for key in f.keys():
        uniprot_id = key.split('_')[-1]
        full_embeddings[uniprot_id] = f[key][:]

# --- Conservation ---
df_cons = pd.read_csv(OUTPUT_DIR / "conservation_scores.csv")
df_core = df_cons[df_cons['gap_fraction'] < 0.5].copy()

# --- Alignment mapping ---
alignment_to_seq = {}
for record in SeqIO.parse(OUTPUT_DIR / "bsh_aligned.fasta", 'fasta'):
    parts = record.id.split('_')
    uniprot_id = parts[-1] if len(parts) > 1 else record.id
    seq = str(record.seq)
    mapping = {}
    seq_pos = 0
    for aln_pos, char in enumerate(seq):
        if char != '-':
            mapping[aln_pos] = seq_pos
            seq_pos += 1
    alignment_to_seq[uniprot_id] = mapping
overlap = set(alignment_to_seq.keys()) & set(per_residue_embeddings.keys())

# --- Activity data ---
df_activity = pd.read_csv(OUTPUT_DIR / "enzyme_amine_activity.csv")
controls = ['CTRL1', 'CTRL2', 'CTRL3', 'CTRL4', 'CTRL5', 'CTRL6', 'CTRL7']
canonical = ['taurine', 'glycine']
df_activity = df_activity[~df_activity['Enzyme'].isin(controls)]
df_activity = df_activity[~df_activity['Amine'].isin(canonical)]
df_activity = df_activity[~df_activity['Amine'].isin(EXCLUDE_AMINES)]
df_activity['Hydroxyl'] = df_activity['Hydroxyl'].replace('3a,7a,12k', '3a7a12k')

# --- Amine SMILES ---
df_smiles = pd.read_excel(DATA_DIR / "bsh_reactants_SMILES_corrected.xlsx")
name_map = {
    '2,3-Diaminopropinoic Acid': '2,3_diaminopropionic acid',
    '2-aminophenol': '2_aminophenol',
    '3-methoxytyramine HCl': '3_methoxytyramine',
    '4-aminophenol': '4_aminophenol',
    'L-Alanine': 'alanine', 'L-Arginine': 'arginine',
    'Asparagine': 'asparagine', 'Cadaverine': 'cadaverine',
    'L-Citrulline': 'citrulline', 'L-Cysteine': 'cysteine',
    'Dopamine HCl': 'dopamine',
    'gamma-Aminobutyric acid >99%': 'gaba',
    'L-Glutamine': 'glutamine', 'Glycyl-L-Valine': 'glyglycine',
    'L-Histidine': 'histidine', 'L-Lysine': 'lysine',
    'L-Methionine': 'methionine',
    'L-Ornithine monohydrochloride': 'ornithine',
    'L-Phenylalanine': 'phenylalanine', 'DL-Proline': 'proline',
    'Putrescine': 'putrescine', 'L-Serine': 'serine',
    'L-Threonine': 'threonine', 'Tryptamine': 'tryptamine',
}
amine_mols = {}
for _, row in df_smiles.iterrows():
    name = row['Compound_Name']
    smiles = row['SMILES']
    norm_name = name_map.get(name, name.lower().replace(' ', '_').replace('-', '_'))
    if pd.isna(smiles): continue
    mol = Chem.MolFromSmiles(smiles.split('.')[0])
    if mol is not None:
        amine_mols[norm_name] = mol

# --- MolT5 ---
molt5_base_path = DATA_DIR / "molt5_base_amine_embeddings.csv"
repr_molt5_base = {}
if molt5_base_path.exists():
    df_m5b = pd.read_csv(molt5_base_path)
    for _, row in df_m5b.iterrows():
        repr_molt5_base[row['amine']] = row.drop('amine').values.astype(np.float32)

# --- Replicate labels ---
df_heatmap_amines = pd.read_csv(DATA_DIR / "NEW_Stage2_BAs_amines_for_heatmap_manual.csv")
df_heatmap_subs = pd.read_csv(DATA_DIR / "NEW_Stage2_BAs_subs_for_heatmap_manual.csv")

def parse_product_col(col):
    parts = col.rsplit('_', 1)
    if len(parts) != 2 or not parts[1].isdigit(): return None, None, None
    product_id = parts[1]; remainder = parts[0]
    known_hydroxyls = ['3a,7a,12k', '3a7a12k', '3a12k', '3a7k', '3k12a', '3k7a', 'Di', 'Mono', 'Tri']
    for h in sorted(known_hydroxyls, key=len, reverse=True):
        if remainder.startswith(h + '_'):
            return h, remainder[len(h)+1:], product_id
    return None, None, None

df_raw = pd.merge(df_heatmap_subs, df_heatmap_amines, on=['filename', 'Code', 'Replicate'], how='outer')
df_raw = df_raw[df_raw['Code'].notna() & (df_raw['Code'] != 'NA')].copy()
meta_cols = ['filename', 'Code', 'Replicate']
product_cols = [c for c in df_raw.columns if c not in meta_cols]
product_info = {}
for col in product_cols:
    h, a, pid = parse_product_col(col)
    if h is not None:
        product_info[col] = {'hydroxyl': h.replace('3a,7a,12k', '3a7a12k'), 'amine': a, 'product_id': pid}

df_long = df_raw.melt(id_vars=['Code', 'Replicate'], value_vars=list(product_info.keys()),
                       var_name='product', value_name='intensity')
df_long['amine'] = df_long['product'].map(lambda x: product_info[x]['amine'])
df_long['hydroxyl'] = df_long['product'].map(lambda x: product_info[x]['hydroxyl'])
df_long['intensity'] = df_long['intensity'].fillna(0)
df_long['detected'] = (df_long['intensity'] > 0).astype(int)

replicate_detection = df_long.groupby(['Code', 'amine', 'hydroxyl', 'Replicate']).agg(
    any_detected=('detected', 'max')).reset_index()
replicate_counts = replicate_detection.groupby(['Code', 'amine', 'hydroxyl']).agg(
    n_reps=('any_detected', 'count'), n_detected=('any_detected', 'sum')).reset_index()
replicate_counts['active_majority'] = (replicate_counts['n_detected'] >= 2).astype(int)

df_activity = df_activity.merge(
    replicate_counts[['Code', 'amine', 'hydroxyl', 'n_detected', 'active_majority']],
    left_on=['Enzyme', 'Amine', 'Hydroxyl'], right_on=['Code', 'amine', 'hydroxyl'], how='left')
df_activity['active_majority'] = df_activity['active_majority'].fillna(
    df_activity['active_approach2'].astype(int)).astype(int)

print("Building features...")

# --- Helper functions ---
def get_nonconserved_embedding(enzyme_id, conservation_threshold=0.6, pooling='max'):
    if enzyme_id not in per_residue_embeddings or enzyme_id not in alignment_to_seq: return None
    embed = per_residue_embeddings[enzyme_id]
    aln_map = alignment_to_seq[enzyme_id]
    variable_aln_positions = df_core[df_core['conservation_score'] < conservation_threshold]['alignment_position'].values
    seq_positions = [aln_map[ap] for ap in variable_aln_positions if ap in aln_map and aln_map[ap] < len(embed)]
    if not seq_positions: return None
    return embed[seq_positions].max(axis=0) if pooling == 'max' else embed[seq_positions].mean(axis=0)

def compute_physicochemical(mol):
    return np.array([
        Descriptors.MolWt(mol), Descriptors.MolLogP(mol), Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol), Descriptors.NumAromaticRings(mol),
        Descriptors.NumAliphaticRings(mol), Descriptors.FractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol), rdMolDescriptors.CalcNumAmideBonds(mol),
        Descriptors.NumValenceElectrons(mol), Descriptors.MaxPartialCharge(mol),
        Descriptors.MinPartialCharge(mol),
        Descriptors.BalabanJ(mol) if Descriptors.BalabanJ(mol) != 0 else 0.0,
    ], dtype=np.float32)

def encode_bile_acid(hydroxyl_value):
    vec = np.zeros(BA_DIM, dtype=np.float32)
    info = PATTERN_INFO.get(hydroxyl_value)
    if info is None: return vec
    c3, c7, c12, n_oh, n_keto = info
    for pos_idx, status in enumerate([c3, c7, c12]):
        base = pos_idx * 3
        if status == 'a': vec[base] = 1.0
        elif status == 'k': vec[base + 1] = 1.0
        else: vec[base + 2] = 1.0
    vec[9] = n_oh; vec[10] = n_keto
    vec[11] = 1.0 if hydroxyl_value not in ('Mono', 'Di', 'Tri') else 0.0
    return vec

def enzyme_holdout_split_seed(X, y, enzymes, seed, test_size=0.2, val_size=0.2):
    enzymes_arr = np.array(enzymes)
    unique_enzymes = np.unique(enzymes_arr)
    profiles = np.array([y[enzymes_arr == e].mean() for e in unique_enzymes])
    bins = pd.cut(profiles, bins=5, labels=False)
    train_val_enz, test_enz = train_test_split(unique_enzymes, test_size=test_size, random_state=seed, stratify=bins)
    tv_profiles = np.array([y[enzymes_arr == e].mean() for e in train_val_enz])
    tv_bins = pd.cut(tv_profiles, bins=5, labels=False)
    train_enz, val_enz = train_test_split(train_val_enz, test_size=val_size, random_state=seed, stratify=tv_bins)
    train_mask = np.isin(enzymes_arr, train_enz)
    val_mask = np.isin(enzymes_arr, val_enz)
    test_mask = np.isin(enzymes_arr, test_enz)
    return {
        'X_train': X[train_mask], 'y_train': y[train_mask],
        'X_val': X[val_mask], 'y_val': y[val_mask],
        'X_test': X[test_mask], 'y_test': y[test_mask],
    }

# --- Build enzyme repr ---
enzyme_repr = {}
d = {}
for eid in overlap:
    emb = get_nonconserved_embedding(eid, conservation_threshold=0.6, pooling='max')
    if emb is not None: d[eid] = emb
enzyme_repr['noncons_max'] = d
enzyme_repr['full_protein'] = {eid: emb for eid, emb in full_embeddings.items()}

# --- Build amine repr ---
amines_needed = df_activity['Amine'].unique()
all_amines_sorted = sorted(amines_needed)
amine_to_idx = {a: i for i, a in enumerate(all_amines_sorted)}
n_amines_onehot = len(all_amines_sorted)

repr_physchem_onehot = {}
for name in amines_needed:
    phys = compute_physicochemical(amine_mols[name]) if name in amine_mols else np.zeros(N_PHYSCHEM, dtype=np.float32)
    onehot = np.zeros(n_amines_onehot, dtype=np.float32)
    if name in amine_to_idx: onehot[amine_to_idx[name]] = 1.0
    repr_physchem_onehot[name] = np.concatenate([phys, onehot])
physchem_onehot_dim = N_PHYSCHEM + n_amines_onehot

molt5_base_dim = len(list(repr_molt5_base.values())[0]) if repr_molt5_base else 0

PCA_DIMS = 50
repr_hybrid = {}
if repr_molt5_base:
    molt5_amines_available = [a for a in amines_needed if a in repr_molt5_base]
    molt5_matrix = np.array([repr_molt5_base[a] for a in molt5_amines_available])
    pca = PCA(n_components=min(PCA_DIMS, molt5_matrix.shape[0], molt5_matrix.shape[1]))
    molt5_pca = pca.fit_transform(molt5_matrix)
    molt5_pca_dict = {a: molt5_pca[i] for i, a in enumerate(molt5_amines_available)}
    actual_pca_dims = molt5_pca.shape[1]
    for name in amines_needed:
        physchem_oh = repr_physchem_onehot[name]
        pca_vec = molt5_pca_dict.get(name, np.zeros(actual_pca_dims, dtype=np.float32)).astype(np.float32)
        repr_hybrid[name] = np.concatenate([physchem_oh, pca_vec])
    hybrid_dim = physchem_onehot_dim + actual_pca_dims

amine_reprs = {'physchem_onehot': (repr_physchem_onehot, physchem_onehot_dim)}
if repr_molt5_base: amine_reprs['molt5_base'] = (repr_molt5_base, molt5_base_dim)
if repr_hybrid: amine_reprs['hybrid'] = (repr_hybrid, hybrid_dim)

# --- Bile acid encoding ---
bile_acid_encodings = {h: encode_bile_acid(h) for h in df_activity['Hydroxyl'].unique()}

# --- Build feature matrices ---
label_schemes = ['active_approach2', 'active_majority']
feature_matrices = {}
for enz_name, enz_dict in enzyme_repr.items():
    for ami_name, (ami_dict, ami_dim) in amine_reprs.items():
        combo_key = f"{enz_name}__{ami_name}"
        X_list, enzymes_list = [], []
        labels = {ls: [] for ls in label_schemes}
        for _, row in df_activity.iterrows():
            enzyme, amine, hydroxyl = row['Enzyme'], row['Amine'], row['Hydroxyl']
            if enzyme not in enz_dict or amine not in ami_dict: continue
            X_list.append(np.concatenate([enz_dict[enzyme], ami_dict[amine], bile_acid_encodings[hydroxyl]]))
            enzymes_list.append(enzyme)
            for ls in label_schemes: labels[ls].append(int(row[ls]))
        feature_matrices[combo_key] = {
            'X': np.array(X_list, dtype=np.float32),
            'labels': {ls: np.array(v, dtype=np.int32) for ls, v in labels.items()},
            'enzymes': enzymes_list,
        }

print("Training models and collecting log loss curves...")

# --- Train and collect curves ---
all_curves = {}
seed = 42

for combo_key, fm in feature_matrices.items():
    X = fm['X']
    enzymes = fm['enzymes']
    for label_scheme in label_schemes:
        y = fm['labels'][label_scheme]
        split = enzyme_holdout_split_seed(X, y, enzymes, seed=seed)
        X_train, y_train = split['X_train'], split['y_train']
        X_val, y_val = split['X_val'], split['y_val']
        X_test, y_test = split['X_test'], split['y_test']

        n_neg = (y_train == 0).sum()
        n_pos = max((y_train == 1).sum(), 1)

        model = xgb.XGBClassifier(scale_pos_weight=n_neg / n_pos, **XGB_PARAMS)
        evals_result = {}
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val), (X_test, y_test)],
            verbose=False,
        )
        evals_result = model.evals_result()

        key = f"{combo_key} | {label_scheme}"
        all_curves[key] = {
            'train': evals_result['validation_0']['logloss'],
            'val': evals_result['validation_1']['logloss'],
            'test': evals_result['validation_2']['logloss'],
            'combo': combo_key,
            'label_scheme': label_scheme,
        }
        n_rounds = len(evals_result['validation_0']['logloss'])
        print(f"  {key}: {n_rounds} rounds, "
              f"train={evals_result['validation_0']['logloss'][-1]:.4f}, "
              f"val={evals_result['validation_1']['logloss'][-1]:.4f}, "
              f"test={evals_result['validation_2']['logloss'][-1]:.4f}")

# --- Plot: 4x3 grid (rows=label×enzyme, cols=amine) ---
combo_keys = list(feature_matrices.keys())
enz_names = list(enzyme_repr.keys())
ami_names = list(amine_reprs.keys())

fig, axes = plt.subplots(4, 3, figsize=(20, 20), sharex=True)

row_configs = [
    ('noncons_max', 'active_approach2'),
    ('noncons_max', 'active_majority'),
    ('full_protein', 'active_approach2'),
    ('full_protein', 'active_majority'),
]

for row_idx, (enz_name, label_scheme) in enumerate(row_configs):
    for col_idx, ami_name in enumerate(ami_names):
        ax = axes[row_idx, col_idx]
        combo_key = f"{enz_name}__{ami_name}"
        key = f"{combo_key} | {label_scheme}"
        curves = all_curves[key]

        rounds = range(1, len(curves['train']) + 1)
        ax.plot(rounds, curves['train'], label='Train', color='#3498db', linewidth=1.5, alpha=0.8)
        ax.plot(rounds, curves['val'], label='Validation', color='#e74c3c', linewidth=1.5, alpha=0.8)
        ax.plot(rounds, curves['test'], label='Test', color='#2ecc71', linewidth=1.5, alpha=0.8)

        # Mark early stopping point
        best_round = np.argmin(curves['val']) + 1
        best_val = min(curves['val'])
        ax.axvline(best_round, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.scatter([best_round], [best_val], color='#e74c3c', s=40, zorder=5, marker='v')

        ax.set_title(f"{enz_name} + {ami_name}\n({label_scheme})", fontsize=10, fontweight='bold')
        ax.set_ylabel('Log Loss', fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

        if row_idx == 3:
            ax.set_xlabel('Boosting Round', fontsize=9)

plt.suptitle('Train / Validation / Test Log Loss Curves\n(seed=42, enzyme hold-out split)',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(PRODUCT_DIR / 'logloss_curves_all.png', dpi=150, bbox_inches='tight')
print(f"\nSaved: {PRODUCT_DIR / 'logloss_curves_all.png'}")

# --- Plot 2: Zoomed overlay of just val curves for comparison ---
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

colors_combo = {
    'noncons_max__physchem_onehot': '#e74c3c',
    'noncons_max__molt5_base': '#c0392b',
    'noncons_max__hybrid': '#e67e22',
    'full_protein__physchem_onehot': '#3498db',
    'full_protein__molt5_base': '#2980b9',
    'full_protein__hybrid': '#8e44ad',
}
linestyles = {
    'physchem_onehot': '-',
    'molt5_base': '--',
    'hybrid': ':',
}

for ax, label_scheme in zip(axes, label_schemes):
    for combo_key in feature_matrices.keys():
        key = f"{combo_key} | {label_scheme}"
        curves = all_curves[key]
        enz_name, ami_name = combo_key.split('__')
        rounds = range(1, len(curves['val']) + 1)
        ax.plot(rounds, curves['val'],
                label=combo_key.replace('__', ' + '),
                color=colors_combo.get(combo_key, 'gray'),
                linestyle=linestyles.get(ami_name, '-'),
                linewidth=1.8, alpha=0.85)

    ax.set_xlabel('Boosting Round', fontsize=11)
    ax.set_ylabel('Validation Log Loss', fontsize=11)
    ax.set_title(f'Validation Log Loss Comparison\n({label_scheme})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PRODUCT_DIR / 'logloss_curves_val_overlay.png', dpi=150, bbox_inches='tight')
print(f"Saved: {PRODUCT_DIR / 'logloss_curves_val_overlay.png'}")

# --- Plot 3: Train-Val gap over rounds ---
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

for ax, label_scheme in zip(axes, label_schemes):
    for combo_key in feature_matrices.keys():
        key = f"{combo_key} | {label_scheme}"
        curves = all_curves[key]
        gap = [v - t for t, v in zip(curves['train'], curves['val'])]
        rounds = range(1, len(gap) + 1)
        enz_name, ami_name = combo_key.split('__')
        ax.plot(rounds, gap,
                label=combo_key.replace('__', ' + '),
                color=colors_combo.get(combo_key, 'gray'),
                linestyle=linestyles.get(ami_name, '-'),
                linewidth=1.8, alpha=0.85)

    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Boosting Round', fontsize=11)
    ax.set_ylabel('Val - Train Log Loss (Gap)', fontsize=11)
    ax.set_title(f'Overfitting Gap Over Rounds\n({label_scheme})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PRODUCT_DIR / 'logloss_gap_over_rounds.png', dpi=150, bbox_inches='tight')
print(f"Saved: {PRODUCT_DIR / 'logloss_gap_over_rounds.png'}")

print("\nDone! All log loss curve figures saved.")
