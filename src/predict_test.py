"""
flips the GENERATIVE model (condition -> sequence) to a classifier (sequence ->
condition). Uses the likelihood-based classification approach:


Entrypoint:
    python src/predict_test.py --config configs/v1.yaml \
        --checkpoint checkpoints/markov_baseline.pkl \
        --output outputs/model_v1_test_predictions.csv

"""

import argparse
import csv
import math
import yaml

from checkpoint import load_checkpoint
from dataset import load_and_split


def softmax(values: list) -> list:
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


def predict_row(model, seq: str, label_to_idx: dict) -> dict:
    """Returns per-condition probabilities (softmax over log-likelihoods)
    and the argmax predicted condition."""
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    log_probs = []
    for idx in range(len(label_to_idx)):
        lp = model.sequence_log_prob(seq, idx)
        log_probs.append(lp if lp == lp else float("-inf"))  # NaN guard -> -inf

    probs = softmax(log_probs)
    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    return {
        "predicted_condition": idx_to_label[best_idx],
        "predicted_probability": probs[best_idx],
        "per_condition_probabilities": {idx_to_label[i]: probs[i] for i in range(len(probs))},
    }


def main(config_path: str, checkpoint_path: str, output_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model, ckpt = load_checkpoint(checkpoint_path)
    label_to_idx = ckpt["label_to_idx"]

    split = load_and_split(cfg)
    test_df = split.test
    if len(test_df) == 0:
        raise SystemExit(f"data.test_csv in {config_path} is empty or not found -- "
                          f"check configs/v1.yaml points at your actual test.csv")

    seq_col, label_col = cfg["data"]["seq_col"], cfg["data"]["label_col"]
    condition_names = sorted(label_to_idx.keys())

    rows = []
    for _, row in test_df.iterrows():
        pred = predict_row(model, row[seq_col], label_to_idx)
        out_row = {
            "gene": row.get("gene", ""),
            "true_condition": row[label_col],
            "predicted_condition": pred["predicted_condition"],
            "predicted_probability": round(pred["predicted_probability"], 6),
        }
        for cond in condition_names:
            out_row[f"prob_{cond}"] = round(pred["per_condition_probabilities"][cond], 6)
        rows.append(out_row)

    fieldnames = ["gene", "true_condition", "predicted_condition", "predicted_probability"] + \
                 [f"prob_{c}" for c in condition_names]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    n_correct = sum(1 for r in rows if r["true_condition"] == r["predicted_condition"])
    print(f"Predicted {len(rows)} test examples, {n_correct}/{len(rows)} correct "
          f"({100*n_correct/len(rows):.1f}% accuracy -- Katelyn's script will compute the full metric set)")
    print(f"Written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v1.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/model_v1_test_predictions.csv")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.output)