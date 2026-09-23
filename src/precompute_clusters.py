"""Write data/derived/clusters_70.csv -- 70%-identity clusters, built once.

cd-hit lives on the head node but not on the compute nodes, so jobs submitted with
sbatch cannot call it. The clustering is deterministic and identical for every
experiment, so it is computed here and read from disk everywhere else.
"""
from pathlib import Path
import sys, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as lb
from splits import identity_matrix, cdhit_merged_clusters

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    enz = set(lb.build(min_reps=1).Enzyme)
    cl = cdhit_merged_clusters(identity=0.70, M=identity_matrix(enz))
    d = pd.DataFrame({"Enzyme": list(cl), "cluster": list(cl.values())})
    d = d.sort_values(["cluster", "Enzyme"])
    out = ROOT / "data/derived/clusters_70.csv"
    d.to_csv(out, index=False)
    print(f"{len(d)} enzymes in {d.cluster.nunique()} clusters -> {out.relative_to(ROOT)}")
