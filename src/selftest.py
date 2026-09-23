"""Check that a clone of this repository is complete and working.

    python src/selftest.py

Verifies that the embeddings, the assay table and the trained model are all
present and consistent, makes a real prediction, and confirms the site is
self-contained. No network and no SLURM needed.
"""
from pathlib import Path
import sys, pickle, numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import embeddings as EM, labels as lb

ok = True


def check(label, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'ok' if cond else 'FAIL'}] {label}{'  ' + detail if detail else ''}")


print("1. protein embeddings")
p = EM.SOURCES["ProstT5"]["path"]
check("ProstT5 embeddings found", p.exists(), str(p).replace(str(ROOT), "<repo>"))
check("bundled with the repository", str(p).startswith(str(ROOT)))

print("\n2. assay labels")
a = lb.build(min_reps=1)
check("built from data/raw", len(a) > 0,
      f"{len(a):,} cells, {int(a.active.sum()):,} active ({a.active.mean():.1%}), "
      f"{a.Enzyme.nunique()} enzymes")
check(f"cut-off is {lb.THRESHOLD:,.0f}", lb.THRESHOLD == 50_000)

print("\n3. trained model")
mp = ROOT / "models/bsh_prostt5_xgb_production.pkl"
check("production model present", mp.exists())
d = pickle.load(open(mp, "rb"))
check("model and labels agree on the rule",
      d["label_rule"]["threshold"] == lb.THRESHOLD,
      f"{d['embedding']}/{d['pooling']}, {d['n_estimators']} rounds, "
      f"trained on {d['trained_on']['enzymes']} enzymes")

print("\n4. a real prediction")
E = EM.load("ProstT5")
acc = sorted(set(E) & set(a.Enzyme))[0] if set(E) & set(a.Enzyme) else sorted(E)[0]
z = np.asarray(E[acc], dtype=np.float32)[None, :]
X = np.hstack([d["scaler"].transform(z).astype(np.float32),
               d["amine_bits"].loc[["gaba"]].to_numpy(np.float32),
               d["core_cols"].loc[["Tri"]].to_numpy(np.float32)]).astype(np.float32)
raw = float(d["model"].predict_proba(X)[:, 1][0])
check("model predicts", 0.0 <= raw <= 1.0,
      f"{acc} + gaba + Tri -> {raw:.4f} raw, "
      f"{float(d['isotonic'].predict([raw])[0]):.4f} calibrated")

print("\n5. the website")
h = (ROOT / "site/index.html").read_text()
check("index.html present", len(h) > 0, f"{len(h)/1e6:.2f} MB")
check("every dataset inlined", h.count("/*__") == 0,
      "open it directly in a browser, no server needed")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
