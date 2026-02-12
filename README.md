# BSH Conjugation Model

Predicting bile salt hydrolase (BSH) enzyme activity on bile acid-amine conjugation pairs using machine learning.

## Background

Bile salt hydrolases (BSH) are microbial enzymes that deconjugate bile acid-amine conjugates in the gut. The Lukowski and Dorrestein labs screened a panel of BSH enzymes against combinations of bile acids and amines to identify which enzymes can catalyze novel conjugation reactions. Product formation was measured by LC-MS intensity across 3 experimental replicates.

This repository contains the data analysis pipeline, replicate quality assessment, and feature engineering for predicting BSH conjugation activity from:

- **Enzyme embeddings** (ProtT5 protein language model, per-residue)
- **Conservation-guided residue selection** (non-conserved positions from MSA)
- **Amine molecular representations** (physicochemical descriptors, one-hot, MolT5)

## Sequence Alignment & Conservation

The notebook `notebooks/bsh_alignment_conservation.ipynb` aligns all 127 BSH sequences with MUSCLE and identifies conserved vs variable positions across the family.

Out of 330 well-occupied alignment columns, only **16 positions are >=95% conserved** -- the catalytic and structural core of the BSH family:

- **C** (Cys) -- catalytic nucleophile
- **H** (His), **D** (Asp) -- complete the Ntn-hydrolase catalytic triad
- **E** (Glu x2), **R** (Arg x2), **N** (Asn x2) -- active site / substrate binding
- **G** (Gly x4), **P** (Pro x2), **T** (Thr) -- structural flexibility and turns

The remaining ~95% of positions are variable, representing the sequence diversity that drives differences in substrate specificity across BSH enzymes.

![Conserved residues analysis](images/conserved_residues_analysis.png)

## Replicate Consistency Analysis

Before building predictive models, we assessed the quality and reproducibility of the LC-MS measurements across 3 experimental replicates (134 enzymes x 94 products).

![Replicate consistency summary](images/replicate_consistency_summary.png)

**Key findings:**
- **GABA** and **asparagine** are the most reliably detected novel amine conjugates (low CV, high detection consistency)
- **2-aminophenol** (40% inconsistent) and **GABA_M+H2O** (43%) are the noisiest -- product appears in only 1-2 of 3 replicates
- **Glycine** and **taurine** (canonical substrates) are among the most reproducible signals
- The gap between "any replicate" vs "all 3 replicates" activity rate reveals which amines have borderline detection

![Detection consistency](images/detection_consistency_per_amine.png)

![Strict activity heatmap](images/amine_bile_acid_heatmap_strict.png)

See `notebooks/amine_replicate_consistency.ipynb` and `notebooks/replicate_consistency_analysis.ipynb` for full analysis.

### Enzyme Replicate Consistency

We extended the replicate consistency analysis to the **enzyme axis** — asking which BSH enzymes give reliable, reproducible signal vs. which are noisy.

![Enzyme consistency distribution](images/enzyme_consistency_distribution.png)

**Key findings:**
- **Enzyme identity is 2x more predictive of inconsistency than amine identity** (point-biserial r = 0.469 vs r = 0.258). The enzyme you pick matters more than the amine for whether a measurement will be reproducible.
- Inconsistency is **heavily skewed**: median enzyme inconsistency is 10.6%, but 28 enzymes (21%) have >25% of their products inconsistently detected, contributing 57% of all inconsistent measurements.
- The noisiest enzymes show a **catastrophic single-replicate failure pattern** — one replicate detects 60+ products while the other two detect only 7 (the baseline substrates). For example, A0A1C7H271 shows rep1=8, rep2=7, rep3=77 (range of 70 products).

![Enzyme ranked inconsistency](images/enzyme_ranked_inconsistency.png)

**Enzyme reliability tiers:**

| Tier | Count | % | Avg Inconsistency | Avg Products Always Detected |
|------|-------|---|-------------------|------------------------------|
| Tier 1: Highly Reliable | 9 | 7% | 3.2% | 30.3 |
| Tier 2: Moderate | 62 | 46% | 8.3% | 29.5 |
| Tier 3: Somewhat Noisy | 35 | 26% | 13.9% | 20.2 |
| Tier 4: Very Noisy | 28 | 21% | 48.1% | 7.4 |

![Enzyme reliability tiers](images/enzyme_reliability_tiers.png)

**Borderline signal:** Inconsistent products have **26x weaker signal** than consistent products (median intensity 7,170 vs 187,278), confirming they sit near the LC-MS detection limit.

![Borderline signal analysis](images/borderline_signal_analysis.png)

**Pairwise replicate agreement:** Per-enzyme detection counts across replicates are well-correlated for most enzymes, but outliers (falling far off the y=x line) correspond to the Tier 4 enzymes with catastrophic replicate failure.

![Enzyme replicate pairwise](images/enzyme_replicate_pairwise.png)

See `notebooks/enzyme_replicate_consistency.ipynb` for full analysis.

## Conservation Threshold Sweep & Max-Pooling Analysis

We tested 8 conservation thresholds x 2 pooling strategies (mean vs max) x 2 models (XGBoost, MLP) x 10 enzyme hold-out seeds = 320 experiments to optimize the enzyme representation.

![Threshold sweep](images/threshold_sweep_lines.png)

**Key findings:**
- **Max pooling consistently outperforms mean pooling** at every threshold (ROC-AUC gap of 0.03-0.05)
- **Threshold 0.6 with max pooling** is the optimal configuration for XGBoost (ROC-AUC: 0.840, PR-AUC: 0.641)
- The max-pool advantage grows with more residues (dilution effect on mean pooling)
- A small set of ~10 alignment positions dominate the max-pool signal -- likely binding pocket residues

![Max contributing residues](images/max_contributing_residues.png)

The top max-contributing position (alignment position 201) accounts for 7.2% of all max contributions across 1024 embedding dimensions, suggesting these positions are functionally important for substrate specificity.

See `notebooks/conservation_threshold_sweep.ipynb` for full analysis.

## Full-Protein Max Pooling Comparison

The earlier conservation threshold sweep only compared mean vs max pooling on **non-conserved residues**. The `full_protein` baseline always used a pre-computed mean-pooled embedding. To make a fair comparison, we ran a 4-way experiment using per-residue embeddings for all strategies:

**4 strategies x 3 models x 10 enzyme hold-out seeds = 120 experiments:**

| Strategy | Residues | Pooling | ROC-AUC | PR-AUC | F1 | LL Gap |
|----------|----------|---------|---------|--------|----|--------|
| full_protein_mean | All | Mean | 0.797 +/- 0.050 | 0.611 +/- 0.087 | 0.605 +/- 0.071 | +0.356 |
| **full_protein_max** | **All** | **Max** | **0.808 +/- 0.052** | **0.622 +/- 0.086** | **0.609 +/- 0.068** | **+0.324** |
| noncons_mean | Non-conserved | Mean | 0.784 +/- 0.047 | 0.604 +/- 0.080 | 0.595 +/- 0.068 | +0.361 |
| noncons_max | Non-conserved | Max | 0.799 +/- 0.034 | 0.606 +/- 0.065 | 0.591 +/- 0.060 | +0.326 |

![Pooling comparison boxplots](images/pooling_comparison_boxplots.png)

**Key findings:**
- **Max pooling improves performance regardless of residue selection** -- both full_protein_max and noncons_max outperform their mean-pooled counterparts
- **Full protein max pooling is the best single strategy** (ROC-AUC 0.808), slightly edging out noncons_max (0.799)
- **The max pooling advantage is consistent across all 3 models** (XGBoost, RF, MLP), not just XGBoost
- **Max pooling reduces overfitting**: the log loss gap is smaller for max-pooled strategies (+0.324) vs mean-pooled (+0.356), suggesting max pooling produces more discriminative features that generalize better
- **Conservation filtering helps reduce variance**: noncons_max has the tightest confidence interval (0.034 std) vs full_protein_max (0.052 std)

![Pooling comparison all models](images/pooling_comparison_all_models.png)

![Pooling comparison overfit gap](images/pooling_comparison_overfit_gap.png)

See `training/full_protein_max_comparison.py` for the full experiment script.

## Product-Level Model: Enzyme + Amine + Bile Acid Core

The earlier models collapsed bile acid hydroxylation patterns (Mono/Di/Tri and positional variants like 3a7k, 3k12a) into a single enzyme-amine pair by aggregating across all bile acid substrates. The **product-level model** (`notebooks/bsh_product_level_model.ipynb`) preserves this hydroxylation specificity, treating each **(Enzyme, Amine, Hydroxylation pattern)** triple as a separate sample.

### Three Feature Blocks

Each sample is represented by three concatenated feature blocks:

| Block | Dimensions | Source | Description |
|-------|-----------|--------|-------------|
| **Enzyme** | 1024 | ProtT5 per-residue embeddings | Two variants tested: `noncons_max` (conservation-guided, non-conserved positions max-pooled) and `full_protein` (mean-pooled whole sequence) |
| **Amine** | 41 / 768 / 91 | RDKit / MolT5 / hybrid | Three variants: `physchem_onehot` (15 physicochemical descriptors + one-hot), `molt5_base` (MolT5 language model, 768-dim), `hybrid` (physchem_onehot + PCA-50 of MolT5) |
| **Bile Acid Core** | 12 | Positional encoding | C3/C7/C12 hydroxylation status (alpha-OH, keto, or unspecified) + hydroxyl count + keto count + is_specific flag |

### Two Label Schemes

Two different approaches were used to define "active" enzyme-amine-hydroxylation triples:

- **`active_approach2` (median-based):** A product is active if its LC-MS intensity exceeds a per-amine epsilon threshold (1000 intensity units) after excluding canonical conjugates (taurine, glycine). This approach was used in all earlier models and captures any detectable signal above noise.

- **`active_majority` (replicate-aware, new):** A product is active only if it was detected in **at least 2 of 3 experimental replicates**. This is more conservative and filters out sporadic/irreproducible detections. Where replicate data was unavailable, the approach2 label was used as fallback. The two schemes disagree on ~X% of samples, with majority vote being stricter.

### Experiment Grid

**120 total experiments:** 2 enzyme representations x 3 amine representations x 2 label schemes x 10 random seeds = 120 models, all evaluated with **enzyme hold-out cross-validation** (stratified 64/16/20 train/val/test split by enzyme activity profile).

**XGBoost configuration:** n_estimators=300, max_depth=3, learning_rate=0.05, reg_alpha=1.0, reg_lambda=5.0, subsample=0.7, colsample_bytree=0.7, min_child_weight=5, early_stopping_rounds=30, scale_pos_weight=auto.

### Results Summary

#### active_approach2 (median-based labels)

| Rank | Enzyme | Amine | Dims | ROC-AUC | PR-AUC | LL Gap |
|------|--------|-------|------|---------|--------|--------|
| 1 | noncons_max | molt5_base | 1804 | 0.776 +/- 0.042 | 0.526 +/- 0.060 | +0.191 |
| 2 | noncons_max | hybrid | 1093 | 0.774 +/- 0.045 | 0.524 +/- 0.068 | +0.182 |
| 3 | noncons_max | physchem_onehot | 1072 | 0.771 +/- 0.041 | 0.518 +/- 0.066 | +0.182 |
| 4 | full_protein | molt5_base | 1804 | 0.742 +/- 0.057 | 0.489 +/- 0.056 | +0.176 |
| 5 | full_protein | physchem_onehot | 1072 | 0.739 +/- 0.063 | 0.484 +/- 0.074 | +0.177 |
| 6 | full_protein | hybrid | 1093 | 0.741 +/- 0.063 | 0.478 +/- 0.067 | +0.180 |

#### active_majority (replicate-aware labels)

| Rank | Enzyme | Amine | Dims | ROC-AUC | PR-AUC | LL Gap |
|------|--------|-------|------|---------|--------|--------|
| 1 | noncons_max | hybrid | 1093 | 0.753 +/- 0.052 | 0.543 +/- 0.059 | +0.232 |
| 2 | full_protein | hybrid | 1093 | 0.752 +/- 0.022 | 0.563 +/- 0.054 | +0.248 |
| 3 | full_protein | physchem_onehot | 1072 | 0.750 +/- 0.019 | 0.571 +/- 0.050 | +0.232 |
| 4 | full_protein | molt5_base | 1804 | 0.746 +/- 0.020 | 0.555 +/- 0.055 | +0.209 |
| 5 | noncons_max | molt5_base | 1804 | 0.749 +/- 0.044 | 0.549 +/- 0.054 | +0.229 |
| 6 | noncons_max | physchem_onehot | 1072 | 0.747 +/- 0.054 | 0.542 +/- 0.066 | +0.223 |

### Feature Importance: Three-Block Analysis

Across all configurations, **enzyme features dominate** (68-94% of total importance), amine features contribute 5-31%, and bile acid core features contribute only 1-2%.

![Three-block feature importance](images/feature_importance_three_blocks.png)

The bile acid positional encoding adds minimal discriminative power -- C12_keto and C3_aOH are the most important bile acid features, but their absolute contribution is small. This suggests that hydroxylation pattern alone does not strongly predict activity beyond what enzyme and amine identity already capture.

![Top features per block](images/top_features_per_block.png)

### Representation Comparisons

**Enzyme representations:** `noncons_max` (conservation-guided) consistently outperforms `full_protein` on ROC-AUC with approach2 labels, confirming that filtering to non-conserved residues removes noise from the conserved structural core. However, with majority labels, the gap narrows.

![Enzyme representation comparison](images/enzyme_representation_comparison.png)

**Amine representations:** All three (physchem_onehot, molt5_base, hybrid) perform comparably. MolT5-base has a marginal edge on PR-AUC, but the differences are within standard deviation. The 768-dim MolT5 embeddings do not meaningfully outperform the 41-dim physicochemical+onehot representation.

![Amine representation comparison](images/amine_representation_comparison.png)

**Label scheme comparison:** The `active_majority` labels yield higher PR-AUC and F1 than `active_approach2`, suggesting the replicate-aware labels reduce noise and make the classification task slightly more learnable. However, they also show larger log loss gaps (more overfitting).

![Label comparison](images/label_comparison.png)

### Log Loss Curves & Overfitting Analysis

The raw train/val/test log loss curves reveal the learning dynamics across all configurations:

![Log loss curves](images/logloss_curves_all.png)

**Key observations:**
- Training loss steadily decreases while validation loss plateaus early, indicating the model exhausts learnable signal quickly
- `full_protein` configs trigger early stopping sooner (51-88 rounds) vs `noncons_max` (95-300 rounds), suggesting the full protein embedding has less extractable signal
- `noncons_max + molt5_base` (approach2) runs all 300 rounds without early stopping, accumulating the largest overfitting gap

![Validation log loss overlay](images/logloss_curves_val_overlay.png)

The validation curves show `full_protein` combos reaching lower absolute validation loss despite worse test set metrics -- a sign of distribution shift between validation and test enzymes in the hold-out splits.

![Overfitting gap over rounds](images/logloss_gap_over_rounds.png)

The train-val gap grows approximately linearly with boosting rounds across all configurations, with no configuration showing signs of convergence. This monotonic gap growth is a hallmark of a model that is memorizing training patterns rather than learning generalizable features.

### Assessment & Current Limitations

**The model is not yet performing well enough for reliable predictions.** Despite systematic optimization of enzyme representations, amine representations, label schemes, and regularization, the model remains in a regime where noise and overfitting dominate.

**Performance plateau:** The product-level model (ROC-AUC ~0.74-0.78, PR-AUC ~0.48-0.57) shows a **notable drop** from the earlier pair-level model (ROC-AUC 0.84, PR-AUC 0.64). This is expected -- predicting activity at the individual hydroxylation pattern level is a harder task with more samples but sparser positive labels. Even the pair-level model at 0.84 ROC-AUC reflects moderate, not strong, predictive power.

**What the log loss curves tell us:**
1. **The model learns fast then overfits.** Most useful signal is extracted in the first 50-80 rounds. After that, additional rounds only memorize training noise.
2. **The overfitting gap never stabilizes.** In a well-calibrated model, you'd expect the gap to plateau. Here it grows linearly, meaning the regularization (L1/L2, subsampling, early stopping) is slowing but not preventing overfitting.
3. **Representations make marginal differences.** Swapping enzyme or amine representations shifts metrics by ~0.03 ROC-AUC at most. This suggests the bottleneck is not in feature engineering but in the fundamental signal-to-noise ratio of the data.

**Likely root causes:**
- **Enzyme noise is the dominant noise source.** The enzyme consistency analysis shows 28 enzymes (21%) are "very noisy" — contributing 57% of all inconsistent measurements. Many of these show catastrophic single-replicate failure (one rep detects 60+ products, others detect 7), suggesting experimental issues rather than true biological variability. This sets a hard noise ceiling on any model trained on this data.
- **Dataset size vs dimensionality:** With ~8,000 product-level samples and 1024+ features, the model is in a high-dimensional regime where XGBoost can easily overfit. The enzyme hold-out evaluation makes this harder since the model must generalize to entirely unseen protein sequences.
- **Bile acid core adds little.** The 12-dim hydroxylation encoding contributes only 1-2% feature importance. This could mean (a) hydroxylation specificity is genuinely driven by enzyme identity rather than the bile acid pattern itself, or (b) the positional encoding doesn't capture the right chemical information about bile acid structure.
- **Label noise ceiling.** With ~17.7% of enzyme-product combinations showing inconsistent detection across replicates (driven more by enzyme identity than amine identity), there is a hard noise floor that limits achievable performance. The majority-vote labels help but don't eliminate this.
- **Data quality issues.** Duplicate enzyme sequences (Q8Y5J3 / A0A3Q0NGD6), missing SMILES for 4 amines, and 10 amines with no detected products all reduce the effective dataset size and introduce noise.

**Potential next directions:**
- **Enzyme reliability weighting** — downweight or exclude the 28 Tier 4 enzymes that contribute the most noise, or use sample weighting by enzyme consistency tier
- **Dimensionality reduction on enzyme embeddings** (PCA to 50-100 dims) to reduce the feature-to-sample ratio
- **Neural network architectures** (attention-based or graph neural networks) that can learn non-linear enzyme-amine-bile acid interactions
- **Data augmentation** through replicate-level training instead of aggregated labels
- **Incorporating 3D structural information** about the bile acid binding pocket rather than sequence-only features
- **Transfer learning** from larger protein-ligand interaction datasets before fine-tuning on BSH-specific data
- **Remove or merge duplicate sequences** and audit which experimental controls are inadvertently included in training

## Repository Structure

```
BSH_model/
├── README.md
├── requirements.txt
├── .gitignore
├── data/                              # Input data files
│   ├── NEW_Stage2_BAs_amines_for_heatmap_manual.csv  # LC-MS intensities: amine products (3 reps)
│   ├── NEW_Stage2_BAs_subs_for_heatmap_manual.csv    # LC-MS intensities: substrate products (3 reps)
│   ├── Seqs_list_total.fasta                         # 127 BSH protein sequences
│   ├── Seqs_list_total.h5                            # ProtT5 whole-protein embeddings
│   ├── bsh_reactants_SMILES_corrected.xlsx           # Corrected reactant SMILES
│   ├── molt5_base_amine_embeddings.csv               # MolT5-base amine embeddings (768-dim)
│   ├── molt5_small_amine_embeddings.csv              # MolT5-small amine embeddings
│   └── swap_enumeration_with_core_smiles.xlsx        # Enumeration with core SMILES
├── notebooks/                         # Analysis notebooks
│   ├── data_preprocessing_BSH_enzyme_model.ipynb     # Data preprocessing pipeline
│   ├── bsh_alignment_conservation.ipynb              # Sequence alignment & conservation
│   ├── replicate_consistency_analysis.ipynb           # Replicate quality assessment
│   ├── amine_replicate_consistency.ipynb              # Per-amine detection consistency
│   ├── enzyme_replicate_consistency.ipynb             # Per-enzyme detection consistency & reliability tiers
│   ├── amine_activity_analysis.ipynb                  # Amine activity profiling
│   ├── amine_representation_comparison.ipynb          # Amine feature comparison
│   ├── molt5_amine_representation.ipynb               # MolT5 molecular embeddings
│   ├── bsh_nonconserved_residue_model.ipynb           # Non-conserved residue model
│   ├── enzyme_amine_combination.ipynb                 # Enzyme + amine representation sweep
│   ├── conservation_threshold_sweep.ipynb             # Conservation threshold optimization
│   ├── bile_acid_hydroxylation_model.ipynb            # Bile acid core analysis
│   ├── bsh_amine_prediction_model.ipynb               # Prediction model notebook
│   ├── bsh_product_level_model.ipynb                   # Product-level model (enzyme + amine + bile acid)
│   ├── plot_logloss_curves.py                          # Log loss curve analysis script
│   ├── amine_umap_explorer.ipynb                      # UMAP visualization
│   └── BSH_conjugation_model.ipynb                    # MIL model training notebook
├── images/                            # Figures for documentation
├── training/                          # Standalone training scripts
└── outputs/                           # Generated by notebooks (gitignored)
```

## Setup

```bash
pip install -r requirements.txt
```

For the alignment notebook, [MUSCLE](https://drive5.com/muscle/) must also be installed and available as `./muscle` in the project root (or on your PATH). On macOS: `brew install brewsci/bio/muscle`.

## Data Files

| File | Description |
|------|-------------|
| `NEW_Stage2_BAs_amines_for_heatmap_manual.csv` | LC-MS intensity for 82 amine products across 134 enzymes x 3 replicates |
| `NEW_Stage2_BAs_subs_for_heatmap_manual.csv` | LC-MS intensity for 12 substrate products (taurine/glycine conjugates) |
| `bsh_reactants_SMILES_corrected.xlsx` | Corrected reactant SMILES for bile acids and amines |
| `Seqs_list_total.fasta` | 127 BSH protein sequences |
| `Seqs_list_total.h5` | ProtT5 whole-protein embeddings |
| `molt5_base_amine_embeddings.csv` | MolT5-base molecular embeddings for amines (768-dim) |
| `molt5_small_amine_embeddings.csv` | MolT5-small molecular embeddings for amines |
| `swap_enumeration_with_core_smiles.xlsx` | Pre-computed enumeration with core SMILES annotations |

## Known Data Discrepancies

### Enzyme count: 134 in activity data vs 127 in FASTA

The activity CSV reports 134 unique enzyme codes, but the FASTA file contains only 127 sequences. The 7 missing entries are **not BSH enzymes**:

| Code | Type |
|------|------|
| `Negative_ctrl_1` | Negative control |
| `Negative_ctrl_2` | Negative control |
| `Negative_ctrl_3` | Negative control |
| `Only_substrate_1` | Substrate-only control |
| `Only_substrate_2` | Substrate-only control |
| `Only_substrate_3` | Substrate-only control |
| `Pencillin_amidase` | Reference enzyme (not a BSH) |

After removing controls, 134 - 7 = 127 BSH enzymes, matching the FASTA exactly.

### Duplicate enzyme sequence

Two FASTA entries share an **identical 325-residue sequence**:
- `Q8Y5J3` (Listeria BSH)
- `A0A3Q0NGD6` (Listeria BSH)

These are the same protein from different database entries. Out of 127 FASTA entries, there are only **126 unique amino acid sequences**. Models trained on these data treat them as separate enzymes, but they provide redundant information.

### Amines/products with missing or incorrect SMILES

Four amines that appear in product columns lack correct SMILES in the reactants file:

| Amine | Issue |
|-------|-------|
| **cystine** | Reactants file has L-Cysteine (monomer), not cystine (disulfide dimer) |
| **glyglycine** | Glycylglycine (Gly-Gly dipeptide) not in reactants file |
| **serotonin** | Not in reactants file |
| **tyramine** | Not in reactants file |

Additionally, `Di_unconjugated_34634` represents a free (unconjugated) bile acid, not an amine conjugate.

### Reactant amines with no detected products

These 10 amines were included in the experimental assay but produced **no detectable conjugation products** in any enzyme:

1,3-Diaminopropane, 5-Aminovaleric Acid, Aspartic Acid, Beta-Alanine, Histamine, Isoniazid, L-Glutathione (Reduced), L-Homoserine, L-Leucine, Glycyl-L-Valine

These amines are absent from all downstream modeling since they have no positive activity labels.

## Key Results

The best-performing model uses a **regularized XGBoost classifier** (max_depth=3, reg_alpha=1.0, reg_lambda=5.0, subsample=0.7) evaluated with **enzyme hold-out cross-validation** (10 random splits, 80/20 train/test) to ensure generalization to unseen enzymes.

**Model input features (1065-dim):**
- **Enzyme (1024-dim):** ProtT5 per-residue embeddings at non-conserved alignment positions (conservation score < 0.6), max-pooled across selected residues. Non-conserved positions are identified from a multiple sequence alignment of all 127 BSH sequences -- these variable positions capture the sequence diversity that drives substrate specificity differences.
- **Amine (41-dim):** 15 RDKit physicochemical descriptors (molecular weight, LogP, TPSA, H-bond donors/acceptors, rotatable bonds, aromatic rings, etc.) + 26-dim one-hot encoding of amine identity. This compact representation outperformed both Morgan fingerprints (1024-dim) and MolT5 molecular language model embeddings (768-dim), likely because the smaller feature space reduces overfitting given the limited training data.

**Prediction target:** Binary classification of whether an enzyme-amine pair produces a detectable deconjugation product (active_approach2 label, aggregated across bile acid cores with max).

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.840 +/- 0.031 |
| PR-AUC | 0.641 +/- 0.061 |
| F1 Score | 0.615 +/- 0.036 |
| Feature importance | 94% enzyme / 6% amine |
| Replicate noise ceiling | ~15% of enzyme-product combos show inconsistent detection across 3 reps |

**Key findings from representation analysis:**
- Max pooling outperforms mean pooling at every conservation threshold (ROC-AUC gap of 0.03-0.05), because a small set of ~10 residue positions dominate the max-pool signal
- The enzyme embedding drives 94% of the model's predictions, while amine features contribute 6% -- suggesting that enzyme identity is far more predictive than amine chemistry
- Compact amine representations (41-dim) reduce overfitting compared to sparse fingerprints (1024-dim), as seen in smaller train-validation log loss gaps
