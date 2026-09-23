"""Score the 66 signal-peptide proteins twice -- full sequence and trimmed -- and compare.

The question is not "are these good BSH candidates". Most of them are enzymes the model
already trained on. The question is whether cutting the N-terminal signal peptide and
re-embedding changes what the model believes about the same protein.

  full     ~/prost5_embeddings/Seqs_list_total_prost5.h5            (the training embeddings)
  trimmed  ~/prost5_embeddings/proteins_without_signal_peptides_trimmed.h5

Both are keyed by entry name (Aureitalea_A0A2S7KLN3); the accession is the last
underscore-separated field, which is how the model's training labels are keyed.

Every protein is scored against all 25 amines x 3 core classes = 75 combinations.
"""
from pathlib import Path
import pickle, h5py, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score

import embeddings as EM, labels as lb, holdout

ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()
OUT = ROOT / "outputs"
def _pick(*c):
    for x in c:
        if Path(x).exists():
            return Path(x)
    return Path(c[0])


EMB = ROOT / "data/embeddings"
H5 = {"full":    _pick(EMB / "Seqs_list_total_prost5.h5",
                       HOME / "prost5_embeddings/Seqs_list_total_prost5.h5"),
      "trimmed": _pick(EMB / "proteins_without_signal_peptides_trimmed.h5",
                       HOME / "prost5_embeddings/proteins_without_signal_peptides_trimmed.h5")}

# which model to score with. The production model saw all 115 enzymes, so for the 58
# candidates that are training enzymes its full-sequence score is recall, not prediction.
# The dev model held 23 of them back, which is what made the earlier comparison
# interpretable. Both are kept; pass --dev to reproduce the original analysis.
MODELS = {"production": ("models/bsh_prostt5_xgb_production.pkl",
                         "candidate_predictions.csv", "candidate_trim_effect.csv"),
          "dev":        ("models/bsh_prostt5_xgb.pkl",
                         "candidate_predictions_devmodel.csv",
                         "candidate_trim_effect_devmodel.csv")}


def read_h5(path):
    """entry name -> 1024-vector. ProstT5 writes int16 holding float16 bit patterns."""
    with h5py.File(path, "r") as f:
        # int16 in the file, bfloat16 bit patterns in fact -- see embeddings._bf16
        return {k: EM._bf16(f[k][:]) for k in f.keys()}


def main(which="production"):
    mpath, pred_csv, eff_csv = MODELS[which]
    d = pickle.load(open(ROOT / mpath, "rb"))
    print(f"model: {mpath}  ({d['trained_on']['enzymes']} enzymes, "
          f"{d['trained_on']['split']}, {d['n_estimators']} rounds)\n")
    E = {k: read_h5(p) for k, p in H5.items()}
    entries = sorted(E["trimmed"])                      # the 66 we were given
    both = [e for e in entries if e in E["full"]]
    only_trimmed = [e for e in entries if e not in E["full"]]
    acc = {e: e.split("_")[-1] for e in entries}

    amines = list(d["amine_bits"].index)
    cores = list(d["core_cols"].index)
    print(f"{len(entries)} proteins x {len(amines)} amines x {len(cores)} cores "
          f"= {len(entries)*len(amines)*len(cores):,} predictions per variant")
    print(f"  {len(both)} have both a full and a trimmed embedding")
    if only_trimmed:
        print(f"  {len(only_trimmed)} trimmed-only (no full embedding to compare): {only_trimmed}")

    # how far did trimming move each protein's vector, before any model is involved
    cos = {e: float(np.dot(E['full'][e], E['trimmed'][e]) /
                    (np.linalg.norm(E['full'][e]) * np.linalg.norm(E['trimmed'][e])))
           for e in both}

    grid = pd.MultiIndex.from_product([entries, amines, cores],
                                      names=["entry", "Amine", "Hydroxyl"]).to_frame(index=False)
    A = d["amine_bits"].loc[grid.Amine].to_numpy(np.float32)
    C = d["core_cols"].loc[grid.Hydroxyl].to_numpy(np.float32)

    for variant in ("full", "trimmed"):
        src = E[variant]
        Z = np.stack([src[e] if e in src else E["trimmed"][e] for e in grid.entry])
        X = np.hstack([d["scaler"].transform(Z).astype(np.float32), A, C]).astype(np.float32)
        raw = d["model"].predict_proba(X)[:, 1]
        grid[f"raw_{variant}"] = raw.round(4)
        grid[f"cal_{variant}"] = d["isotonic"].predict(raw).round(4)

    grid["accession"] = grid.entry.map(acc)
    grid["delta"] = (grid.raw_trimmed - grid.raw_full).round(4)
    grid["comparable"] = grid.entry.isin(both)
    grid["embedding_cosine"] = grid.entry.map(cos).round(4)

    # ground truth where we have it -- 58 of these are enzymes the model was trained on
    act = lb.build(min_reps=1)
    truth = act.set_index(["Enzyme", "Amine", "Hydroxyl"]).active
    grid["measured"] = [truth.get((a, am, h), np.nan)
                        for a, am, h in zip(grid.accession, grid.Amine, grid.Hydroxyl)]
    assign = holdout.assignment()
    grid["split"] = grid.accession.map(assign)

    cols = ["entry", "accession", "split", "Amine", "Hydroxyl", "measured",
            "raw_full", "raw_trimmed", "delta", "cal_full", "cal_trimmed",
            "embedding_cosine", "comparable"]
    grid[cols].to_csv(OUT / pred_csv, index=False)

    c = grid[grid.comparable]
    print(f"\n{'='*70}\nHOW MUCH DOES TRIMMING MOVE THE PREDICTIONS?\n{'='*70}")
    print(f"  embedding cosine similarity (full vs trimmed vector):")
    v = pd.Series(cos)
    print(f"     min {v.min():.4f}   median {v.median():.4f}   max {v.max():.4f}")
    print(f"     below 0.99: {(v < 0.99).sum()} proteins")
    print(f"\n  prediction agreement over {len(c):,} comparable combinations:")
    print(f"     Pearson r   {np.corrcoef(c.raw_full, c.raw_trimmed)[0,1]:.4f}")
    print(f"     Spearman    {c[['raw_full','raw_trimmed']].corr('spearman').iloc[0,1]:.4f}")
    print(f"     mean |delta|  {c.delta.abs().mean():.4f}")
    print(f"     median |delta| {c.delta.abs().median():.4f}")
    print(f"     max |delta|   {c.delta.abs().max():.4f}")
    flip = ((c.raw_full >= .5) != (c.raw_trimmed >= .5))
    print(f"     calls that flip across 0.5: {flip.sum():,} of {len(c):,} ({flip.mean():.1%})")

    m = c.dropna(subset=["measured"])
    if len(m):
        y = m.measured.astype(int).to_numpy()
        print(f"\n{'='*70}\nWHICH VARIANT AGREES BETTER WITH THE MEASUREMENTS?\n{'='*70}")
        print(f"  {len(m):,} combinations with a measured answer "
              f"({m.accession.nunique()} enzymes, {y.mean():.1%} active)")
        print(f"  NOTE: most of these enzymes are in the model's training set, so these")
        print(f"        numbers are inflated for BOTH variants. Only the gap is meaningful.\n")
        for variant in ("full", "trimmed"):
            p = m[f"raw_{variant}"].to_numpy()
            print(f"     {variant:8s} PR-AUC {average_precision_score(y, p):.4f}   "
                  f"ROC-AUC {roc_auc_score(y, p):.4f}   "
                  f"accuracy at 0.5 {((p >= .5).astype(int) == y).mean():.4f}")

    print(f"\n{'='*70}\nPROTEINS MOST CHANGED BY TRIMMING\n{'='*70}")
    per = (c.groupby("entry")
             .agg(cut_cosine=("embedding_cosine", "first"),
                  mean_abs_delta=("delta", lambda s: s.abs().mean()),
                  max_abs_delta=("delta", lambda s: s.abs().max()),
                  flips=("delta", lambda s: 0), n=("delta", "size")))
    per["flips"] = c.groupby("entry").apply(
        lambda g: ((g.raw_full >= .5) != (g.raw_trimmed >= .5)).sum())
    per = per.sort_values("mean_abs_delta", ascending=False)
    print(per.head(12).round(4).to_string())
    per.round(4).to_csv(OUT / eff_csv)
    print(f"\n-> outputs/{pred_csv}  ({len(grid):,} rows)")
    print(f"-> outputs/{eff_csv}   (per protein)")


if __name__ == "__main__":
    import sys
    main("dev" if "--dev" in sys.argv else "production")
