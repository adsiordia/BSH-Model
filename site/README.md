# BSH Conjugation Explorer

Eight tabs: Start here · Trends · Sequence & preference · Enzymes · Amines ·
Amine chemistry · Full map · New predictions · How it was made.

A ninth, "Model vs data", was removed -- it listed individual disagreements between
the out-of-fold predictions and the measurements, which duplicated what the Full map
already shows when toggled to the model's view.

A self-contained page. `index.html` embeds all of its data, so it works from
`file://` — you can copy that one file anywhere and double-click it.

## Serving it from this machine

    ./serve.sh              # start on 127.0.0.1:8811
    ./serve.sh status
    ./serve.sh stop

It binds to loopback only, so it is not exposed to the network. Reach it from
your own computer with an SSH tunnel:

    ssh -N -L 8811:localhost:8811 adsiordia@gpu.jinich.ucsd.edu

Leave that running (it will look like it has hung — that is correct) and open
<http://localhost:8811>.

## Sending it to someone

Copy the single file; no server needed at their end:

    scp adsiordia@gpu.jinich.ucsd.edu:~/BSH-Model-v2/site/index.html .

## Rebuilding after the model or data changes

    python ../src/export_site_data.py    # activity + predictions  -> data.json
    python ../src/export_amine_chem.py   # structures + properties -> chem.json
    python ../src/export_trends.py       # trends + sequence corr  -> trends.json
    python ../src/export_methods.py      # how-it-was-made record  -> methods.json
    python ../src/export_trimming.py     # trimmed vs untrimmed    -> trimming.json
    python ../src/export_candidates.py   # new predictions         -> candidates.json
    python ../src/export_pooling.py      # residue-selection test  -> pooling.json
    python ../src/build_conjugates.py    # product structures      -> conjugates.json
    python build.py                      # inline all six          -> index.html

`build.py` is the only step that touches `index.html`; the six exports are
independent and can be run individually.

`export_candidates.py` builds the "New predictions" tab and needs:

    outputs/candidate_predictions.csv                       src/predict_candidates.py
    data/predictions_data/candidate_organisms.csv           src/fetch_organisms.py

`build_conjugates.py` draws the predicted product for each amine on each degree class --
one representative bile acid per class with its hydroxyls highlighted, rather than every
isomer, since isomers weigh the same and the data cannot distinguish them. That keeps it
to 75 structures (~1.9 MB), small enough to inline, so `index.html` remains self-contained.

`export_pooling.py` covers section 10 and needs:

    outputs/pooling_sweep_compact.csv        src/sweep_pooling.py --compact
    outputs/pooling_sweep_compact_oof.npz
    data/derived/msa_linsi.fasta             src/align.py
    data/derived/conservation.csv            src/conservation.py
    data/derived/alignment_residue_index.csv

`src/align.py` rebuilds the MAFFT alignment and records the command in
`msa_linsi.log`; the older `bsh_aligned.fasta` had no provenance and is superseded.

`fetch_organisms.py` queries UniProt once and caches the raw TSV next to its output;
pass `--refresh` to re-query. 44 of the 66 accessions are deleted entries in current
UniProt, so their organism comes from the entry-name mnemonic, and mnemonics starting
with a digit resolve only to a group, never a species.

`export_methods.py` reads the run artifacts rather than restating them, so the
"How it was made" tab cannot drift away from the code. It needs:

    outputs/sweep_metrics.csv     16 rows: 4 embeddings x 4 algorithms
    outputs/sweep_oof.npz         out-of-fold predictions, for the curves
    outputs/sweep_meta.csv        the rows those predictions line up with
    outputs/logloss_curves.npz    train/validation/held-out traces per round
    data/derived/holdout_split.csv

It also picks up the production model and its locked-test-set result when these
exist:

    models/bsh_prostt5_xgb.pkl        src/train_final.py
    outputs/final_model.json          src/train_final.py
    outputs/test_final_metrics.csv    src/evaluate_final.py
    outputs/test_final_predictions.csv

`export_trimming.py` covers section 9 and needs:

    outputs/candidate_predictions.csv                      src/predict_candidates.py
    models/bsh_prostt5_xgb_production.pkl                  src/train_production.py
    data/predictions_data/signaling_peptide/signal_peptide_manifest.csv
    ~/prost5_embeddings/Seqs_list_total_prost5.h5          (full sequences)
    ~/prost5_embeddings/proteins_without_signal_peptides_trimmed.h5

Regenerate the sweep artifacts with `src/sweep_embeddings.py` and
`src/logloss_curves.py` respectively. `src/predict_candidates.py` scores with the production model
by default; pass `--dev` to reproduce the earlier comparison using the model that
held 23 enzymes back. `src/evaluate_final.py` spends the locked test set — it appends to `outputs/test_set_evaluations.log` so repeated runs are
visible in the record. Thresholds, split parameters and embedding paths are read live
from `labels.py`, `holdout.py` and `embeddings.py`.
