"""Sequence-identity-aware splitting, so near-duplicate enzymes cannot leak
between train / validation / test.

Enzymes are clustered by sequence identity (cd-hit), then whole CLUSTERS are
assigned to splits. Splitting on enzyme id alone is not enough: this panel
contains one pair with 100% identity and ~4% of pairs above 70%.
"""

import itertools
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

ROOT = Path(__file__).resolve().parent.parent
RAW, DERIVED = ROOT / "data/raw", ROOT / "data/derived"

# cd-hit requires the word size to match the identity threshold
_WORD = [(0.7, 5), (0.6, 4), (0.5, 3), (0.4, 2)]


def _word_size(c):
    for lo, n in _WORD:
        if c >= lo:
            return n
    raise ValueError("cd-hit cannot cluster below 40% identity; use alignment_clusters()")


def sequences(enzymes=None):
    """enzyme id -> ungapped sequence."""
    out = {}
    for rec in SeqIO.parse(RAW / "Seqs_list_total.fasta", "fasta"):
        out[rec.id.split("_")[-1]] = str(rec.seq).replace("-", "")
    return {k: v for k, v in out.items() if enzymes is None or k in enzymes}


def cdhit_clusters(enzymes=None, identity=0.5):
    """Cluster with cd-hit; returns {enzyme: cluster_id}."""
    seqs = sequences(enzymes)
    with tempfile.TemporaryDirectory() as td:
        fa, out = Path(td) / "in.fa", Path(td) / "out"
        fa.write_text("".join(f">{k}\n{v}\n" for k, v in seqs.items()))
        subprocess.run(
            ["cd-hit", "-i", str(fa), "-o", str(out), "-c", str(identity),
             "-n", str(_word_size(identity)), "-M", "2000", "-d", "0"],
            check=True, capture_output=True)
        clusters, cid = {}, -1
        for line in (out.with_suffix(".clstr")).read_text().splitlines():
            if line.startswith(">Cluster"):
                cid = int(line.split()[1])
            elif ">" in line:
                clusters[line.split(">")[1].split("...")[0]] = cid
    return clusters


def identity_matrix(enzymes=None):
    """Pairwise % identity from the MSA (ungapped positions only)."""
    aln = {r.id.split("_")[-1]: str(r.seq) for r in SeqIO.parse(DERIVED / "bsh_aligned.fasta", "fasta")}
    if enzymes is not None:
        aln = {k: v for k, v in aln.items() if k in enzymes}
    names = sorted(aln)
    M = np.eye(len(names))
    for i, j in itertools.combinations(range(len(names)), 2):
        a, b = aln[names[i]], aln[names[j]]
        both = [(x, y) for x, y in zip(a, b) if x != "-" and y != "-"]
        M[i, j] = M[j, i] = (sum(x == y for x, y in both) / len(both)) if both else 0.0
    return pd.DataFrame(M, index=names, columns=names)


def split_by_cluster(clusters, seed=0, test_frac=0.2, val_frac=0.2):
    """Assign whole clusters to train/val/test. Returns {enzyme: 'train'|'val'|'test'}.

    Clusters are shuffled and greedily filled so the enzyme counts land near the
    requested fractions -- cluster sizes are uneven, so exact fractions are not
    achievable without breaking clusters.
    """
    rng = np.random.default_rng(seed)
    by_cluster = {}
    for e, c in clusters.items():
        by_cluster.setdefault(c, []).append(e)
    cids = list(by_cluster)
    rng.shuffle(cids)

    n = len(clusters)
    n_test = test_frac * n
    n_val = val_frac * (1 - test_frac) * n
    # train first, so the largest cluster lands in train rather than inflating test
    targets = {"train": n - n_test - n_val, "val": n_val, "test": n_test}
    assign, counts = {}, {k: 0 for k in targets}
    for c in sorted(cids, key=lambda c: -len(by_cluster[c])):   # big clusters first
        # greatest absolute deficit, so the biggest split absorbs the biggest cluster
        split = max(counts, key=lambda k: targets[k] - counts[k])
        for e in by_cluster[c]:
            assign[e] = split
        counts[split] += len(by_cluster[c])
    return assign



CLUSTER_FILE = DERIVED / "clusters_70.csv"


def load_clusters(enzymes=None, identity=0.70):
    """enzyme -> cluster id, from data/derived/clusters_70.csv when it exists.

    cd-hit is installed on the head node but not on the compute nodes, so anything
    submitted with sbatch cannot rebuild the clustering. It is deterministic and the
    same for every experiment, so src/precompute_clusters.py writes it once and this
    reads it. Falls back to computing it when the file is absent and cd-hit is there.
    """
    if CLUSTER_FILE.exists() and identity == 0.70:
        d = pd.read_csv(CLUSTER_FILE)
        cl = dict(zip(d.Enzyme, d.cluster))
        if enzymes is None:
            return cl
        missing = set(enzymes) - set(cl)
        if missing:
            raise KeyError(f"{len(missing)} enzymes absent from {CLUSTER_FILE.name} "
                           f"-- rerun src/precompute_clusters.py: {sorted(missing)[:5]}")
        return {e: cl[e] for e in enzymes}
    return cdhit_merged_clusters(enzymes, identity=identity)


def balanced_group_split(groups, test_size=0.2, seed=0):
    """Split whole clusters in two, balanced by how many ROWS each side gets.

    sklearn's train_test_split over unique cluster ids balances the number of
    CLUSTERS, which is meaningless here: dev cluster sizes run from 1 to 36 enzymes
    and 41 of 48 are singletons. Taking 20% of clusters took 48% of the enzymes in
    two folds, leaving the stopping check larger than the training set.

    Whole clusters still move together, so the no-leakage guarantee is unchanged --
    only the proportions are fixed. Largest clusters are placed first, each going to
    whichever side is furthest below its target share.

    Returns (train_index, holdout_index) into `groups`.
    """
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    order = np.argsort(-counts)                      # largest first
    rng = np.random.default_rng(seed)
    # break size ties randomly so different seeds give different splits
    key = np.lexsort((rng.random(len(uniq)), -counts))
    order = key
    target = test_size * counts.sum()
    hold, n_hold, n_train = set(), 0, 0
    for i in order:
        c, n = uniq[i], counts[i]
        # deficit against each side's target, relative to what this cluster would add
        if n_hold + n / 2 <= target and (n_hold / max(target, 1)) <= (
                n_train / max(counts.sum() - target, 1)):
            hold.add(c); n_hold += n
        else:
            n_train += n
    if not hold:                                     # never return an empty check set
        hold.add(uniq[order[-1]]); n_hold = counts[order[-1]]
    m = np.isin(groups, list(hold))
    return np.where(~m)[0], np.where(m)[0]


def check_leakage(assign, identity=None, enzymes=None):
    """Max sequence identity between any pair that straddles two splits."""
    M = identity_matrix(enzymes) if identity is None else identity
    names = [e for e in M.index if e in assign]
    worst, rows = 0.0, []
    for a, b in itertools.combinations(names, 2):
        if assign[a] != assign[b]:
            v = M.loc[a, b]
            if v > worst:
                worst = v
            if v > 0.4:
                rows.append((a, b, assign[a], assign[b], v))
    return worst, sorted(rows, key=lambda r: -r[-1])


def alignment_clusters(enzymes=None, identity=0.4, M=None):
    """Single-linkage clusters from the MSA identity matrix.

    Connected components of the graph "identity >= threshold". This GUARANTEES
    that no two enzymes in different clusters exceed the threshold -- unlike
    cd-hit, whose greedy heuristic and local-alignment identity definition can
    leave highly similar pairs in separate clusters.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    M = identity_matrix(enzymes) if M is None else M
    A = csr_matrix((M.values >= identity).astype(int))
    n, lab = connected_components(A, directed=False)
    return dict(zip(M.index, lab))


def cdhit_merged_clusters(enzymes=None, identity=0.70, M=None):
    """cd-hit clusters, then merge any that are joined by a >= identity pair.

    cd-hit alone leaves highly similar pairs in separate clusters (its greedy
    representative-based assignment does not chain). This takes its output and
    merges any two cd-hit clusters connected by a pair at or above the threshold,
    restoring the guarantee that no cross-cluster pair exceeds it -- while keeping
    cd-hit's finer partition wherever that is already safe.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    M = identity_matrix(enzymes) if M is None else M
    base = cdhit_clusters(set(M.index), identity=identity)
    ids = sorted(set(base.values()))
    pos = {c: i for i, c in enumerate(ids)}
    A = np.zeros((len(ids), len(ids)), dtype=int)
    names = list(M.index)
    for i, j in itertools.combinations(range(len(names)), 2):
        if M.values[i, j] >= identity:
            a, b = pos[base[names[i]]], pos[base[names[j]]]
            if a != b:
                A[a, b] = A[b, a] = 1
    _, lab = connected_components(csr_matrix(A), directed=False)
    remap = {c: lab[pos[c]] for c in ids}
    return {e: remap[c] for e, c in base.items()}
