"""Representation comparison: does the pretrained Enformer representation carry more condition-relevant signal
than the simple learned embedding your baseline already uses?

Design choice: rather than retraining the full generative model twice
(slow, confounds representation quality with generation-specific training
noise), this trains a small linear/MLP PROBE on top of each frozen
representation to predict the condition label, then compares probe
validation accuracy. 

Entrypoint (once Enformer is actually runnable on lab hardware):
    python src/compare_representations.py --config configs/baseline.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from .seed import set_seed
    from .dataset import load_and_split, SequenceDataset
    from .representations import SimpleEmbeddingRepresentation, get_enformer_embedding
except ImportError:
    from seed import set_seed
    from dataset import load_and_split, SequenceDataset
    from representations import SimpleEmbeddingRepresentation, get_enformer_embedding


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_conditions: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_conditions)

    def forward(self, x):
        return self.fc(x)


def extract_simple_features(seqs_idx: torch.Tensor, rep_model: SimpleEmbeddingRepresentation) -> torch.Tensor:
    with torch.no_grad():
        emb = rep_model(seqs_idx)          # (B, L, D)
        return emb.mean(dim=1)              # mean-pool -> (B, D)


def extract_enformer_features(sequences: list) -> torch.Tensor:
    """NOTE: slow -- one Enformer forward pass per sequence. For a real run,
    batch this and/or subsample sequences per condition (Enformer inference
    cost is the main reason this stays a separate offline step rather than
    living inside the main training loop)."""
    feats = [get_enformer_embedding(seq) for seq in sequences]
    return torch.stack(feats)


def train_probe(features: torch.Tensor, labels: torch.Tensor, num_conditions: int,
                 epochs: int = 30, lr: float = 0.01) -> LinearProbe:
    probe = LinearProbe(features.shape[1], num_conditions)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits = probe(features)
        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()
    return probe


@torch.no_grad()
def probe_accuracy(probe: LinearProbe, features: torch.Tensor, labels: torch.Tensor) -> float:
    preds = probe(features).argmax(dim=-1)
    return (preds == labels).float().mean().item()


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])

    split = load_and_split(cfg)
    label_to_idx = split.label_to_idx
    num_conditions = len(label_to_idx)

    train_ds = SequenceDataset(split.train, label_to_idx, cfg)
    val_ds = SequenceDataset(split.val, label_to_idx, cfg)

    train_seq_idx = torch.stack([train_ds[i][0] for i in range(len(train_ds))])
    train_labels = torch.stack([train_ds[i][1] for i in range(len(train_ds))])
    val_seq_idx = torch.stack([val_ds[i][0] for i in range(len(val_ds))])
    val_labels = torch.stack([val_ds[i][1] for i in range(len(val_ds))])

    print("=== Simple embedding representation ===")
    simple_rep = SimpleEmbeddingRepresentation(embedding_dim=cfg["model"]["embedding_dim"])
    train_feats_simple = extract_simple_features(train_seq_idx, simple_rep)
    val_feats_simple = extract_simple_features(val_seq_idx, simple_rep)
    probe_simple = train_probe(train_feats_simple, train_labels, num_conditions)
    acc_simple = probe_accuracy(probe_simple, val_feats_simple, val_labels)
    print(f"Validation probe accuracy (simple embedding): {acc_simple:.4f}")

    print("\n=== Enformer representation ===")
    try:
        train_seqs = split.train[cfg["data"]["seq_col"]].tolist()
        val_seqs = split.val[cfg["data"]["seq_col"]].tolist()
        train_feats_enf = extract_enformer_features(train_seqs)
        val_feats_enf = extract_enformer_features(val_seqs)
        probe_enf = train_probe(train_feats_enf, train_labels, num_conditions)
        acc_enf = probe_accuracy(probe_enf, val_feats_enf, val_labels)
        print(f"Validation probe accuracy (Enformer embedding): {acc_enf:.4f}")

        print(f"\n=== Result: {'Enformer' if acc_enf > acc_simple else 'Simple'} representation "
              f"performed better ({acc_enf:.4f} vs {acc_simple:.4f}) ===")
    except ImportError as e:
        print(f"\nSkipped Enformer comparison: {e}")
        print("Run on lab hardware with enformer-pytorch installed and a GPU available.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    main(args.config)
