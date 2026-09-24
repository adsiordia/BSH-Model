# BSH Conjugation Model

Predicting which bile salt hydrolases (BSH) attach which amines to which bile acid cores.

BSH enzymes hydrolyse the glycine or taurine amide at C24 of a bile acid and can
re-amidate the core with a different amine. This repository holds the assay labels,
the model, every negative result we could measure, and an interactive site that
presents all of it.

## Quick start

```bash
git clone git@github.com:adsiordia/BSH-Model.git && cd BSH-Model
python src/selftest.py          # confirms the clone is complete and working
```

**1. Look at the results.** Open `site/index.html` in a browser. Every dataset is
inlined, so it needs no server, no network and no Python — it can be opened from disk
or emailed as a single file. Ten tabs cover the assay, the trends, the model, how it
was built and the predictions. To serve it instead:

```bash
cd site && ./serve.sh            # http://localhost:8811
```

**2. Use the trained model.** `models/bsh_prostt5_xgb_production.pkl` is the complete
model, trained on all 115 enzymes. The pickle carries its own scaler, isotonic
calibrator, amine fingerprints, core columns and the exact label rule it was trained
under, so it is self-describing:

```python
import pickle, numpy as np
d = pickle.load(open("models/bsh_prostt5_xgb_production.pkl", "rb"))
d["label_rule"]        # {'threshold': 50000.0, 'min_reps': 1, 'n_reps': 3}
X = np.hstack([d["scaler"].transform(embedding[None, :]),
               d["amine_bits"].loc[["gaba"]].to_numpy("float32"),
               d["core_cols"].loc[["Tri"]].to_numpy("float32")]).astype("float32")
raw = d["model"].predict_proba(X)[:, 1]
cal = d["isotonic"].predict(raw)      # calibrated probability
```

**3. Retrain from scratch.** Everything needed is in the repository — the assay table,
the ProstT5 embeddings and the curated SMILES:

```bash
sbatch jobs_retrain.sh           # or: python src/train_production.py
```

## The data

115 enzymes x 25 amines x 3 bile acid core classes = **8,625 combinations**, all
measured. Amines and bile acids were each pooled into a single master mix, so every
enzyme met every combination and the panel is complete by construction. Every enzyme
was confirmed present by proteomics, so an absent product means the enzyme was there
and did not make it.

A product counts as **made** when its intensity exceeds **50,000** in at least one of
three replicates: **1,381 active cells (16.0%)**.

The cut-off was raised from 10,000 on 2026-09-22. The old value came from negative
controls, but only 9 of 1,458 control measurements are non-zero and the two that set
the bound fall on just 2 of 55 products. 50,000 instead rests on detection
reproducibility across all 2,467 non-zero cells — the share of detections that
reappear in a second replicate climbs 34% (10-25k) to 66% (25-50k) to 82% (50-100k)
and then stays flat. A five-way comparison found **no measurable difference in model
performance** between 10,000, 25,000 and 50,000, so the choice rests on the evidence
behind the label, not on a model gain.

Several LC-MS features can map to one product; 21 of 49 products were detected under
2-4 feature ids. Features are collapsed by taking the **maximum within each
replicate** before thresholding. Summing instead would change 20 cells (0.35%).

## The model

`models/bsh_prostt5_xgb_production.pkl` — XGBoost, 825 rounds, trained on all 115
enzymes. Features are three blocks:

| block | how |
|---|---|
| enzyme | ProstT5 whole-protein embedding, mean-pooled over residues, standardised |
| amine  | Morgan fingerprint, radius 2, 512 bits |
| core   | one-hot over Mono / Di / Tri (counts hydroxyls, not positions) |

Cluster-grouped 5-fold CV, where clusters are connected components of a >=70%
identity graph so no near-duplicate spans a fold. These are cross-validated numbers,
not a held-out test:

| metric | value | chance |
|---|---|---|
| PR-AUC | 0.7642 | 0.1601 |
| **PR-AUC, normalised** | **0.7193** | **0** |
| ROC-AUC | 0.9341 | 0.5 |
| log loss | 0.2371 (0.2093 after isotonic calibration) | 0.4398 |
| Brier | 0.0596 (on calibrated predictions) | — |
| **within-substrate AUC** | **0.6362** | **0.5** |

**Quote the normalised PR-AUC when comparing across splits or label rules.** Raw
PR-AUC has a floor equal to the base rate, so a number measured where 16.0% of cells
are active is not comparable with one measured where 13.9% are. The normalised form
is (PR - base) / (1 - base): 0 is chance, 1 is perfect, whatever the prevalence.
ROC-AUC is already base-rate independent and needs no adjustment.

Within-substrate AUC is averaged over substrate groups holding at least 5 positives
and 5 negatives. Including the sparse groups too gives 0.572 over all 48 — the
filtered figure is the meaningful one, but recomputing naively will not reproduce it.

### Read the last row first

PR-AUC and ROC-AUC are inflated by an effect that has nothing to do with the enzymes.
Ablation over the three feature blocks:

| features | PR-AUC |
|---|---|
| enzyme only | 0.258 |
| amine + core only | 0.738 |
| everything | 0.780 |

**The protein contributes +0.042.** Most of the apparent accuracy is the model
learning which *products* are commonly made — gaba+Tri is produced by 86% of
enzymes, so predicting it is right 86% of the time whichever enzyme you name.

**Within-substrate AUC isolates the enzyme**: fix the amine and core, then ask whether
the right enzymes rank on top. It floors at 0.5. At **0.636** the model has real but
modest power to tell enzymes apart. Quote this number for "which protein should I
test", and the PR-AUC for "which product type will be made".

### Ceiling

Detection reproducibility plateaus at **82%**, and the two identical sequences in the
panel agree only **82.7%** of the time. No model can exceed that. Measured precision
is already within a few points of it.

## Negative results

Kept deliberately — each cost real compute and each constrains what to try next.

- **Protein language models are statistically tied.** ProstT5, ProtT5, ESM-2 650M and
  ESM-3 over 4 algorithms: best-vs-second paired difference +0.004, CI crossing zero,
  p = 0.32.
- **No pooling scheme beats whole-protein mean.** 7 schemes x 4 embeddings, including
  mean+max and alignment-column selection of conserved and non-conserved residues.
  All 28 runs land between 0.270 and 0.298 log loss. One scheme looked like a winner
  on ProstT5 (+0.032, p < 0.001) and did not replicate on the other three.
- **No activity threshold is measurably better.** 5 label rules, every confidence
  interval spanning zero over 22 common substrate groups.
- **Trimming signal peptides degrades a model trained on full sequences.** Feeding
  trimmed embeddings to this model drops 110 real measured products and adds 79 false
  calls. The change is regression to the mean (r = -0.932 between starting score and
  movement; spread collapses 61%), not recovered activity. Dropped products are rare
  ones (median 6.1% prevalence); kept ones are common (49.6%).

## Layout

```
src/             analysis and training code; every script runs standalone
src/selftest.py  verifies a clone is complete and working
jobs_*.sh        SLURM submission scripts (all compute runs through sbatch)
data/raw/        the proteomics table, sequences, SMILES, signal peptides
data/embeddings/ ProstT5 embeddings: what the production model trains and predicts on
data/derived/    MSA, conservation, clusters, holdout split  (see PROVENANCE.md)
models/          trained models as pickles, label rule recorded inside each
outputs/         out-of-fold predictions, sweep results, candidate predictions
site/            the interactive site: template.html + JSON -> build.py -> index.html
images/          figures
```

## The site

```bash
cd site && ./serve.sh          # http://localhost:8811
python build.py               # rebuild index.html from template + JSON
```

`index.html` is self-contained — every dataset is inlined, so it can be opened from
disk or emailed. Ten tabs cover the assay, the trends, the model, how it was built,
and the predictions.

## Reproducing

```bash
python src/precompute_clusters.py   # identity clusters -> data/derived/clusters_70.csv
python src/align.py                 # MAFFT L-INS-i MSA
python src/conservation.py          # per-column occupancy and conservation
sbatch jobs_retrain.sh              # train the production model
sbatch jobs_sweep.sh                # embedding x algorithm sweep
sbatch jobs_regen.sh                # regenerate every site JSON
```

Run compute through SLURM, not on the head node, and cap `n_jobs` to
`SLURM_CPUS_PER_TASK` rather than `-1`.

### Files not in this repository

Two per-residue embedding files exceed GitHub's 100 MB limit and are gitignored:

| file | size | needed by |
|---|---|---|
| `data/raw/esm3_per_residue.h5` | 227 MB | `sweep_pooling.py` only |
| `data/raw/Seqs_list_total_per_residue.h5` | 170 MB | `sweep_pooling.py` only |

Regenerate them by embedding `data/raw/Seqs_list_total.fasta` with ESM-3 and ProstT5
keeping per-residue output. **Nothing else depends on them** — training, prediction
and the site all use the mean-pooled embeddings in `data/embeddings/`, which are
included. ESM-2 embeddings are likewise not bundled and are needed only to reproduce
the four-way embedding comparison, whose result was that the four are statistically
tied.

## Open items

- `src/labels.py` documents `MIN_REPS = 2`, but every caller passes `min_reps=1`
  (1,381 active vs 997). The looser rule is what ran; the discrepancy is unresolved
  and is stated on the site.
- The locked-test figures were measured **before** the cut-off moved to 50,000 and have
  not been re-run. Quote **ROC-AUC 0.924** from that evaluation: it is base-rate
  independent, so it survives both that caveat and the next one intact. PR-AUC was
  0.69 raw, **0.6194 normalised** against its own base rate of 0.1843.
- The dev/test split is grouped by sequence cluster but **not stratified on activity**.
  The test side is somewhat less active (13.9% of cells vs 16.5% in dev), which shifts
  the PR-AUC floor between them. At the enzyme level the difference is not significant
  (median breadth 9 vs 12, Mann-Whitney p = 0.214, n = 23), so the split was left as
  it is rather than invalidating the one held-out measurement — but any future split
  should stratify clusters by breadth before assigning them.
- Predictions on signal-peptide-trimmed sequences come from a model trained on
  untrimmed ones. Settling whether the mature sequence predicts better requires
  retraining on trimmed embeddings for all 115 enzymes.
