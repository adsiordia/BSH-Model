"""Generate per-residue ProtT5 embeddings from a FASTA file.

This reproduces how data/raw/Seqs_list_total_per_residue.h5 was made: each protein
is embedded ONCE, in full. Selecting non-conserved positions happens afterwards,
downstream -- the sequence is never truncated before embedding, because ProtT5 was
trained on natural sequences and a chopped one is out of distribution.

Output: one HDF5 dataset per protein, shape (sequence_length, 1024), float32.

    python src/embed_prott5.py --fasta data/raw/Seqs_list_total.fasta \
                               --out data/raw/per_residue_new.h5

Model is ~2.5 GB on first run (cached in ~/.cache/huggingface afterwards).
On CPU this takes minutes for ~127 short sequences; on a GPU, seconds.
NOTE: this machine has RTX 4090s but a CPU-only torch build. To use them:
    pip install torch --index-url https://download.pytorch.org/whl/cu124
"""
import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from Bio import SeqIO
from transformers import T5EncoderModel, T5Tokenizer

MODEL = "Rostlab/prot_t5_xl_half_uniref50-enc"
DIM = 1024


def preprocess(seq):
    """ProtT5 expects space-separated residues; rare/ambiguous ones map to X."""
    return " ".join(re.sub(r"[UZOB]", "X", str(seq).upper().replace("-", "")))


def embed(fasta, out, batch_size=4, device=None, max_len=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    records = [(r.id, str(r.seq).replace("-", "")) for r in SeqIO.parse(fasta, "fasta")]
    if max_len:
        records = [(i, s) for i, s in records if len(s) <= max_len]
    records.sort(key=lambda r: -len(r[1]))          # longest first: stable memory use
    print(f"{len(records)} sequences, lengths {min(len(s) for _, s in records)}"
          f"-{max(len(s) for _, s in records)}")

    tok = T5Tokenizer.from_pretrained(MODEL, do_lower_case=False, legacy=True)
    model = T5EncoderModel.from_pretrained(MODEL).to(device).eval()
    if device == "cpu":
        model = model.float()                       # half precision is GPU-only

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f, torch.no_grad():
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            enc = tok.batch_encode_plus([preprocess(s) for _, s in batch],
                                        add_special_tokens=True, padding="longest",
                                        return_tensors="pt")
            ids = enc["input_ids"].to(device)
            mask = enc["attention_mask"].to(device)
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
            for j, (name, seq) in enumerate(batch):
                # drop padding and the trailing </s>; keep exactly len(seq) rows
                emb = hidden[j, :len(seq)].cpu().numpy().astype(np.float32)
                assert emb.shape == (len(seq), DIM), f"{name}: {emb.shape} != {(len(seq), DIM)}"
                f.create_dataset(name, data=emb)
            print(f"  {min(i+batch_size, len(records)):>4}/{len(records)}", end="\r")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-len", type=int, default=None)
    a = ap.parse_args()
    embed(a.fasta, a.out, a.batch_size, a.device, a.max_len)
