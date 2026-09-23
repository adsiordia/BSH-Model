"""Fixed three-way split: development (train+validation) and a locked test set.

The test clusters are chosen ONCE, written to data/derived/holdout_split.csv, and
read from that file thereafter. Nothing in development reads the test rows.

Design:
  TEST  ~20% of sequence clusters -- untouched until a single final evaluation
  DEV   the rest, used for cross-validation, tuning and every design decision
          within DEV, 5-fold cluster-grouped CV gives train/validation each fold

Splitting is by CLUSTER (connected components at 70% MSA identity), not by enzyme:
two different enzymes at 95% identity are near-duplicates, so an enzyme-level split
would still leak. check() verifies no cross-split pair exceeds the threshold.
"""
from pathlib import Path
import numpy as np, pandas as pd

from splits import identity_matrix, alignment_clusters, check_leakage

ROOT = Path(__file__).resolve().parent.parent
SPLIT_FILE = ROOT / "data/derived/holdout_split.csv"
IDENTITY = 0.70
TEST_FRACTION = 0.20
SEED = 20260916


def make(enzymes, identity=IDENTITY, test_fraction=TEST_FRACTION, seed=SEED, force=False):
    """Create the split once. Refuses to overwrite unless force=True."""
    if SPLIT_FILE.exists() and not force:
        raise FileExistsError(f"{SPLIT_FILE} already exists -- refusing to reshuffle. "
                              "Pass force=True only if you intend to invalidate it.")
    M = identity_matrix(set(enzymes))
    clusters = alignment_clusters(M=M, identity=identity)
    by_cluster = {}
    for e, c in clusters.items():
        by_cluster.setdefault(c, []).append(e)

    # A test set drawn from one big cluster would measure performance on a single
    # homology group. Clusters larger than the whole test budget are therefore kept
    # in dev, and the test set is accumulated from randomly ordered smaller clusters
    # so it spans many families.
    rng = np.random.default_rng(seed)
    target = test_fraction * len(clusters)
    eligible = [c for c in by_cluster if len(by_cluster[c]) <= target]
    rng.shuffle(eligible)
    test_clusters, n = set(), 0
    for c in eligible:
        if n >= target:
            break
        test_clusters.add(c); n += len(by_cluster[c])

    rows = [{"Enzyme": e, "cluster": c, "split": "test" if c in test_clusters else "dev"}
            for c, members in by_cluster.items() for e in members]
    df = pd.DataFrame(rows).sort_values(["split", "cluster", "Enzyme"])
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SPLIT_FILE, index=False)
    return df


def load():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"{SPLIT_FILE} missing -- run holdout.make() once.")
    return pd.read_csv(SPLIT_FILE)


def assignment():
    """enzyme -> 'dev' | 'test'."""
    d = load()
    return dict(zip(d.Enzyme, d.split))


def clusters():
    """enzyme -> cluster id (needed for grouped CV inside dev)."""
    d = load()
    return dict(zip(d.Enzyme, d.cluster))


def check():
    """Verify the dev/test boundary holds."""
    d = load()
    M = identity_matrix(set(d.Enzyme))
    worst, offenders = check_leakage(assignment(), M, set(d.Enzyme))
    return worst, offenders


if __name__ == "__main__":
    import dataset as ds
    emb = ds.enzyme_embeddings()
    import labels as lb
    enzymes = sorted(set(lb.build().Enzyme) & set(emb))
    if not SPLIT_FILE.exists():
        make(enzymes)
        print(f"created {SPLIT_FILE}")
    d = load()
    print(f"\n{len(d)} enzymes in {d.cluster.nunique()} clusters at {IDENTITY:.0%} identity\n")
    print(d.groupby("split").agg(enzymes=("Enzyme", "size"),
                                 clusters=("cluster", "nunique")).to_string())
    worst, off = check()
    print(f"\n  max sequence identity across the dev/test boundary: {worst:.1%}")
    print(f"  pairs above 40% straddling the boundary: {len(off)}")
    print(f"  -> {'PASS' if worst < IDENTITY else 'FAIL'} (must be below {IDENTITY:.0%})")
