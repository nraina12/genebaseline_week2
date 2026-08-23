"""Entrypoint: python src/train.py --config configs/v1.yaml

Handles both model types, and both data-loading modes (random-split toy
data or Tisha's pre-split gene-locked V1 data, via dataset.py's
load_and_split). When training the LSTM on imbalanced data (V1's 62.5%
KRAS_MAPK_ERK), set train.use_class_balancing: true in the config to
oversample minority conditions via WeightedRandomSampler.
"""

import argparse
import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from seed import set_seed, seed_worker
from dataset import load_and_split, SequenceDataset, compute_sample_weights
from model import build_model
from checkpoint import save_checkpoint


def train_lstm(model, cfg, train_loader, val_loader, device, label_to_idx):
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=4)
    best_val = float("inf")
    patience = 0
    ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], "lstm_best.pt")

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for step, (seq_idx, cond_idx) in enumerate(train_loader):
            seq_idx, cond_idx = seq_idx.to(device), cond_idx.to(device)
            inputs, targets = seq_idx[:, :-1], seq_idx[:, 1:]
            logits = model(inputs, cond_idx)
            loss = loss_fn(logits.reshape(-1, 4), targets.reshape(-1).clamp(max=3))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % cfg["train"]["log_every"] == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

        val_loss = evaluate_lstm_loss(model, val_loader, loss_fn, device)
        print(f"epoch {epoch} val_loss {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            save_checkpoint(model, cfg, epoch, val_loss, label_to_idx, ckpt_path)
        else:
            patience += 1
            if patience >= cfg["train"]["early_stop_patience"]:
                print("Early stopping.")
                break


@torch.no_grad()
def evaluate_lstm_loss(model, loader, loss_fn, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for seq_idx, cond_idx in loader:
        seq_idx, cond_idx = seq_idx.to(device), cond_idx.to(device)
        inputs, targets = seq_idx[:, :-1], seq_idx[:, 1:]
        logits = model(inputs, cond_idx)
        loss = loss_fn(logits.reshape(-1, 4), targets.reshape(-1).clamp(max=3))
        total += loss.item()
        n += 1
    return total / max(n, 1)


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = load_and_split(cfg)
    label_to_idx = split.label_to_idx
    num_conditions = len(label_to_idx)
    print(f"train/val/test sizes: {len(split.train)}/{len(split.val)}/{len(split.test)}, "
          f"{num_conditions} conditions: {list(label_to_idx.keys())}")
    print(f"train condition counts: {dict(split.train[cfg['data']['label_col']].value_counts())}")

    model = build_model(num_conditions, cfg)

    if cfg["model"]["type"] == "markov":
        train_seqs = split.train[cfg["data"]["seq_col"]].tolist()
        train_conds = split.train[cfg["data"]["label_col"]].map(label_to_idx).tolist()
        # oversample minority-condition rows before fitting, if requested --
        # Markov fit() has no notion of sample weights, so this duplicates
        # minority rows to roughly balance condition representation instead
        if cfg["train"].get("use_class_balancing"):
            import random as _random
            rng = _random.Random(cfg["seed"])
            weights = compute_sample_weights(split.train, cfg["data"]["label_col"], label_to_idx)
            max_w = max(weights)
            oversampled_seqs, oversampled_conds = [], []
            for seq, cond, w in zip(train_seqs, train_conds, weights):
                repeats = max(1, round(w / max_w * 5))  # cap oversampling factor at ~5x
                oversampled_seqs.extend([seq] * repeats)
                oversampled_conds.extend([cond] * repeats)
            print(f"class balancing: {len(train_seqs)} -> {len(oversampled_seqs)} rows after oversampling")
            train_seqs, train_conds = oversampled_seqs, oversampled_conds
        # Call the class method directly: ``nn.Module`` may expose a tensor
        # attribute named ``fit`` on the model instance.
        # Resolve the class method dynamically so static type checkers do not
        # assume that ``model`` is a ConditionalLSTM.
        getattr(type(model), "fit")(model, train_seqs, train_conds)
        ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], "markov_baseline.pkl")
        save_checkpoint(model, cfg, epoch=0, val_loss=float("nan"), label_to_idx=label_to_idx, path=ckpt_path)
        print(f"Markov baseline fit and saved to {ckpt_path}")
        return

    train_ds = SequenceDataset(split.train, label_to_idx, cfg)
    val_ds = SequenceDataset(split.val, label_to_idx, cfg)
    g = torch.Generator().manual_seed(cfg["seed"])

    if cfg["train"].get("use_class_balancing"):
        sample_weights = compute_sample_weights(split.train, cfg["data"]["label_col"], label_to_idx)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights),
                                         replacement=True, generator=g)
        train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], sampler=sampler,
                                   worker_init_fn=seed_worker)
        print("class balancing: using WeightedRandomSampler to oversample minority conditions")
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                                   worker_init_fn=seed_worker, generator=g)

    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)

    getattr(model, "to")(device)
    train_lstm(model, cfg, train_loader, val_loader, device, label_to_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    main(args.config)
