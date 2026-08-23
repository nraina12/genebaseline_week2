"""Entrypoint:
    python src/compare_to_real.py \
        --generated outputs/generated_XXXX.jsonl \
        --real data/real_sequences.csv \
        --pwms data/jaspar_pwms.txt \
        --threshold 0.75
"""

import argparse
import csv
import json
import time
import os
from collections import defaultdict

from motif_metrics import load_jaspar_pfms, summarize_sequence
from diversity_metrics import percent_unique, pairwise_similarity_stats, count_near_duplicates

#identifies relevant motifs per condition 
RELEVANT_MOTIFS_BY_CONDITION = {
    "KRAS_MAPK_ERK": ["MA0028.2", "MA0099.3"],   # ETS (ELK1) + AP-1 (FOS::JUN)
    "HNF4G_FOXA1": ["MA0484.1", "MA0148.4"],
    "GATA6": ["MA1104.1"],
    "PTF1A_NEGATIVE": ["MA1619.1"],
}


def load_generated(path: str) -> dict:
    """Returns {condition: [sequence, ...]}"""
    by_cond = defaultdict(list)
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            cond = rec.get("condition") or "unknown"
            by_cond[cond].append(rec["sequence"])
    return by_cond


def load_real(path: str) -> dict:
    by_cond = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            by_cond[row["condition"]].append(row["sequence"])
    return by_cond


def summarize_group(sequences: list, condition: str, pwms: dict, threshold: float) -> dict:
    if not sequences:
        return {"n_sequences": 0}

    motif_results = [
        summarize_sequence(seq, condition, pwms, threshold, RELEVANT_MOTIFS_BY_CONDITION)
        for seq in sequences
    ]
    freqs = [r["relevant_motif_frequency_per_100bp"] for r in motif_results]
    strengths = [r["relevant_motif_strength"]["mean_normalized_score"]
                 for r in motif_results if r["relevant_motif_strength"]["mean_normalized_score"] is not None]

    return {
        "n_sequences": len(sequences),
        "mean_motif_frequency_per_100bp": round(sum(freqs) / len(freqs), 3) if freqs else None,
        "mean_motif_strength": round(sum(strengths) / len(strengths), 3) if strengths else None,
        "percent_unique": round(percent_unique(sequences), 2),
        "pairwise_similarity_mean": round(pairwise_similarity_stats(sequences)["mean"] or 0, 3) if len(sequences) > 1 else None,
        "near_duplicates": count_near_duplicates(sequences) if len(sequences) > 1 else 0,
    }


def main(generated_path: str, real_path: str, pwm_path: str, threshold: float):
    pwms = load_jaspar_pfms(pwm_path)
    generated = load_generated(generated_path)
    real = load_real(real_path)

    all_conditions = sorted(set(generated.keys()) | set(real.keys()))
    comparison = {}

    header = f"{'condition':18s} {'source':10s} {'n':>4s} {'freq/100bp':>11s} {'strength':>9s} {'%unique':>8s}"
    print(header)
    print("-" * len(header))

    for cond in all_conditions:
        comparison[cond] = {}
        for source_name, data in [("generated", generated), ("real", real)]:
            seqs = data.get(cond, [])
            summary = summarize_group(seqs, cond, pwms, threshold)
            comparison[cond][source_name] = summary
            if summary["n_sequences"] > 0:
                print(f"{cond:18s} {source_name:10s} {summary['n_sequences']:>4d} "
                      f"{summary.get('mean_motif_frequency_per_100bp', 'n/a'):>11} "
                      f"{summary.get('mean_motif_strength', 'n/a'):>9} "
                      f"{summary.get('percent_unique', 'n/a'):>8}")
            else:
                print(f"{cond:18s} {source_name:10s}    0  (no sequences)")

    out_dir = os.path.dirname(generated_path) or "outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"comparison_{int(time.time())}.json")
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nFull detail written to {out_path}")

    print("\nNote: PTF1A_NEGATIVE should show LOW freq/strength for its own motif and,")
    print("more importantly, generated tumor-condition sequences (KRAS_MAPK_ERK,")
    print("HNF4G_FOXA1, GATA6) should NOT show elevated PTF1A_NEGATIVE motif activity --")
    print("that's the specificity test from the framework doc. Check per_motif_match_counts")
    print("in the individual motif_metrics.py output for the PTF1A cross-condition check.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", required=True, help=".jsonl from generate.py")
    parser.add_argument("--real", required=True, help=".csv from prepare_real_data.py")
    parser.add_argument("--pwms", required=True, help="JASPAR PFM file")
    parser.add_argument("--threshold", type=float, default=0.75)
    args = parser.parse_args()
    main(args.generated, args.real, args.pwms, args.threshold)
