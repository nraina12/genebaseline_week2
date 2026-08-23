import argparse
import yaml
import torch
import numpy as np

from seed import set_seed
from dataset import load_and_split, SequenceDataset
from checkpoint import load_checkpoint
from model import ConditionalMarkovModel, ConditionalLSTM


def gc_content(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / max(len(seq), 1)


def evaluate_markov(model: ConditionalMarkovModel, split, label_to_idx, cfg):
    test_df = split.test
    log_probs = []
    for _, row in test_df.iterrows():
        cond_idx = label_to_idx[row[cfg["data"]["label_col"]]]
        lp = model.sequence_log_prob(row[cfg["data"]["seq_col"]], cond_idx)
        if lp == lp:  # not NaN
            log_probs.append(lp)
    print(f"Test set mean log-likelihood: {np.mean(log_probs):.3f} (n={len(log_probs)})")

    print("\nPer-condition GC content (real test sequences):")
    for label, idx in label_to_idx.items():
        seqs = test_df[test_df[cfg["data"]["label_col"]] == label][cfg["data"]["seq_col"]]
        if len(seqs) == 0:
            continue
        gc = np.mean([gc_content(s) for s in seqs])
        print(f"  {label}: {gc:.3f} (n={len(seqs)})")


def main(config_path: str, checkpoint_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])

    split = load_and_split(cfg)

    if cfg["model"]["type"] == "markov":
        model, ckpt = load_checkpoint(checkpoint_path)
        if model is not None:
            evaluate_markov(model, split, ckpt["label_to_idx"], cfg)
    else:
        print("LSTM evaluation: extend this with held-out cross-entropy / "
              "perplexity using evaluate_lstm_loss from train.py — same idea, "
              "just called on the test_loader instead of val_loader.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)
