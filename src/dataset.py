"""Load labeled sequences and produce train/val/test splits.

Two modes, both going through the same downstream interface:

1. RANDOM SPLIT (toy data / early testing): one CSV, this module splits
   it randomly using the config seed. This is what data.csv_path +
   load_and_split() always did.

2. PRE-SPLIT / GENE-LOCKED (Tisha's Dataset V1): separate train/val/test
   CSVs that are ALREADY split by her, gene-locked (no gene appears in
   more than one split). Do NOT re-split these -- re-splitting at the row
   level would destroy the gene-locking and let the same gene's multiple
   promoters leak across splits. Set data.train_csv / data.val_csv /
   data.test_csv in the config to use this path instead of data.csv_path.
"""

from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset

from encoding import sequence_to_indices


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    label_to_idx: dict


def _build_label_map(*dfs: pd.DataFrame, label_col: str) -> dict:
    labels = sorted(set().union(*[set(df[label_col].unique()) for df in dfs]))
    return {lab: i for i, lab in enumerate(labels)}


def load_presplit(cfg: dict) -> SplitData:
    """Loads Tisha-style pre-split, gene-locked CSVs directly -- no
    resplitting, no shuffling across splits. Warns if any gene appears in more than one split, 
    and if any sequences are outside the configured min_len/max_len range."""
    import os
    d = cfg["data"]
    train_df = pd.read_csv(d["train_csv"])

    def _load_optional(key: str) -> pd.DataFrame:
        path = d.get(key)
        if not path or not os.path.exists(path):
            print(f"NOTE: {key} ('{path}') not found yet -- using empty placeholder. "
                  f"Fine for now, but training/eval that needs this split won't work until it arrives.")
            return pd.DataFrame(columns=train_df.columns)
        return pd.read_csv(path)

    val_df = _load_optional("val_csv")
    test_df = _load_optional("test_csv")

    seq_col, label_col = d["seq_col"], d["label_col"]

    # sanity check: gene-locking should mean no overlap, if a gene column
    # is present -- warn (don't crash) if that invariant seems violated,
    # since this data comes from an external source we don't control.
    if "gene" in train_df.columns and "gene" in val_df.columns:
        overlap = set(train_df["gene"]) & set(val_df["gene"])
        if overlap:
            print(f"WARNING: {len(overlap)} gene(s) appear in both train and val "
                  f"-- gene-locking may have been violated: {sorted(overlap)[:5]}...")

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(df) == 0:
            continue
        bad_len = df[~df[seq_col].str.len().between(d.get("min_len", 1), d.get("max_len", 10**9))]
        if len(bad_len) > 0:
            print(f"WARNING: {len(bad_len)} {name} sequence(s) outside configured "
                  f"min_len/max_len -- check data.min_len/max_len match this dataset "
                  f"(V1 sequences are 1200bp, not the 200-500bp toy-data range)")

    label_to_idx = _build_label_map(train_df, val_df, test_df, label_col=label_col)
    return SplitData(train=train_df, val=val_df, test=test_df, label_to_idx=label_to_idx)


def load_and_split(cfg: dict) -> SplitData:
    d = cfg["data"]
    if d.get("train_csv"):
        return load_presplit(cfg)

    df = pd.read_csv(d["csv_path"])
    df = df[df[d["seq_col"]].str.len().between(d["min_len"], d["max_len"])].reset_index(drop=True)

    labels = sorted(df[d["label_col"]].unique())
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    df = df.sample(frac=1.0, random_state=cfg["seed"]).reset_index(drop=True)

    n = len(df)
    n_train = int(n * d["train_frac"])
    n_val = int(n * d["val_frac"])

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    return SplitData(train=train_df, val=val_df, test=test_df, label_to_idx=label_to_idx)


def compute_class_weights(df: pd.DataFrame, label_col: str, label_to_idx: dict) -> torch.Tensor:
    """Inverse-frequency class weights, for handling the KRAS_MAPK_ERK
    imbalance Tisha flagged (62.5% of V1). Use with a WeightedRandomSampler
    (see train.py) or as loss weighting."""
    counts = df[label_col].value_counts()
    weights = torch.ones(len(label_to_idx))
    for label, idx in label_to_idx.items():
        weights[idx] = 1.0 / max(counts.get(label, 1), 1)
    weights = weights / weights.sum() * len(label_to_idx)  # normalize to mean 1
    return weights


def compute_sample_weights(df: pd.DataFrame, label_col: str, label_to_idx: dict) -> List[float]:
    """Per-ROW weights for a WeightedRandomSampler -- oversamples minority
    condition rows so each condition is seen roughly equally often during
    training, despite the underlying class imbalance."""
    counts = df[label_col].value_counts()
    return [1.0 / counts[row_label] for row_label in df[label_col]]


class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_to_idx: dict, cfg: dict):
        self.df = df.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.seq_col = cfg["data"]["seq_col"]
        self.label_col = cfg["data"]["label_col"]
        self.pad_to = cfg["data"]["pad_to"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[i]
        seq_idx = sequence_to_indices(row[self.seq_col], self.pad_to)
        cond_idx = torch.tensor(self.label_to_idx[row[self.label_col]], dtype=torch.long)
        return seq_idx, cond_idx
