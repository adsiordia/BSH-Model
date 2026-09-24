"""Does the molecular representation of the amine matter?

Two regimes:
  A. enzyme-grouped CV  -- every amine is seen in training. A one-hot code is a
     perfect lookup, so chemistry can only match it, never beat it.
  B. leave-one-amine-out -- the test amine is UNSEEN. One-hot carries no
     information at all, so this is the only setting where representation counts.
"""
from pathlib import Path
import sys, json, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT/"src"))
import labels as lb, embeddings as EM, splits
pd.set_option("display.width", 200)
N_JOBS = int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", 4))

lab = lb.build(min_reps=1)
chem = json.load(open(ROOT/"site/chem.json"))["amines"]
E = EM.load("ProstT5")
lab = lab[lab.Enzyme.isin(E)].copy()
amines = sorted(lab.Amine.unique()); cores = sorted(lab.Hydroxyl.unique())
print(f"{lab.Enzyme.nunique()} enzymes, {len(amines)} amines, {len(cores)} cores, {len(lab):,} cells")

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512)
def morgan(s):
    m = Chem.MolFromSmiles(s)
    return np.array(gen.GetFingerprint(m), dtype=np.float32) if m is not None else np.zeros(512, np.float32)
DESC = ["mw","logp","tpsa","hbd","hba","rotb","arom","heavy"]
REPR = {}
REPR["Morgan 512"]  = {a: morgan(chem[a]["smiles"]) for a in amines}
REPR["one-hot"]     = {a: np.eye(len(amines), dtype=np.float32)[i] for i, a in enumerate(amines)}
d = np.array([[chem[a]["props"][k] for k in DESC] for a in amines], np.float32)
d = StandardScaler().fit_transform(d)
REPR["8 descriptors"] = {a: d[i] for i, a in enumerate(amines)}
REPR["Morgan + desc"] = {a: np.concatenate([REPR["Morgan 512"][a], REPR["8 descriptors"][a]]) for a in amines}

Z = np.stack([E[e] for e in lab.Enzyme]); Z = StandardScaler().fit_transform(Z).astype(np.float32)
C = pd.get_dummies(lab.Hydroxyl)[cores].to_numpy(np.float32)
y = lab.active.to_numpy(int)
cl = splits.load_clusters(); groups = lab.Enzyme.map(cl).to_numpy()

def model():
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=1, reg_lambda=5,
        scale_pos_weight=max((y==0).sum()/(y==1).sum(),1.0), tree_method="hist",
        n_jobs=N_JOBS, eval_metric="logloss", verbosity=0)

print("\n" + "="*70)
print("A.  enzyme-grouped CV  (every amine seen in training)")
print("="*70)
print(f"  {'representation':16s} {'dims':>5s} {'PR-AUC':>8s} {'ROC-AUC':>8s}")
res = {}
for name, R in REPR.items():
    A = np.stack([R[a] for a in lab.Amine])
    X = np.hstack([Z, A, C]).astype(np.float32)
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = model().fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:,1]
    res[name] = oof
    print(f"  {name:16s} {A.shape[1]:5d} {average_precision_score(y,oof):8.4f} "
          f"{roc_auc_score(y,oof):8.4f}")

print("\n" + "="*70)
print("B.  leave-one-amine-out  (test amine never seen in training)")
print("="*70)
print(f"  {'representation':16s} {'PR-AUC':>8s} {'ROC-AUC':>8s}   vs base rate")
base = y.mean()
for name, R in REPR.items():
    A = np.stack([R[a] for a in lab.Amine])
    X = np.hstack([Z, A, C]).astype(np.float32)
    oof = np.zeros(len(y))
    for a in amines:
        te = (lab.Amine == a).to_numpy(); tr = ~te
        if y[tr].sum() == 0: continue
        m = model().fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:,1]
    pr, rc = average_precision_score(y, oof), roc_auc_score(y, oof)
    print(f"  {name:16s} {pr:8.4f} {rc:8.4f}   {pr/base:6.2f}x")
print(f"\n  base rate (random) PR-AUC = {base:.4f}")
