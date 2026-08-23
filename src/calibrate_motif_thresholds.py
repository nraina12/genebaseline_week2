"""Fixes the "HNF4G matches everywhere" problem: one fixed threshold
(e.g. 0.75) applied to every motif treats them as if they had identical
score distributions, which they don't -- some PWMs are "sharper" (high
information content, hard to score well by chance) and some are "looser"
(low information content, easy to clear a fixed bar on essentially random
sequence). A loose motif at a global 0.75 threshold matches constantly and
tells you nothing.

This script computes, for each motif independently, what score it
achieves against BACKGROUND sequence (sequence with no reason to actually
contain that motif) and sets the threshold at a chosen percentile of that
background distribution -- e.g. "only count it as a match if it scores
higher than 99% of what background sequence would score by chance."

Two background sources, combined:
  1. Random ACGT sequences -- cheap, no data dependency.
  2. Dinucleotide-shuffled real sequences (if a reference set is given) --
     preserves realistic base composition while destroying real motif
     structure, which is a stronger background than pure random.

Entrypoint:
    python src/calibrate_motif_thresholds.py \
        --pwms data/jaspar_pwms.txt \
        --reference data/real_sequences.csv \
        --percentile 99 \
        --output data/motif_thresholds.json
"""

import argparse
import csv
import json
import random
from typing import Optional

from motif_metrics import load_jaspar_pfms, score_all_windows

BASES = "ACGT"


def random_background(n_sequences: int, length: int, seed: int = 0) -> list:
    rng = random.Random(seed)
    return ["".join(rng.choice(BASES) for _ in range(length)) for _ in range(n_sequences)]


def dinucleotide_shuffle(seq: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    if len(seq) < 4:
        chars = list(seq)
        rng.shuffle(chars)
        return "".join(chars)
    dinucs = [seq[i:i + 2] for i in range(0, len(seq) - 1, 2)]
    rng.shuffle(dinucs)
    shuffled = "".join(dinucs)
    if len(seq) % 2 == 1:
        shuffled += seq[-1]
    return shuffled


def percentile(values: list, pct: float) -> float:
    if not values:
        return 1.0  # no background data -> maximally strict fallback
    sorted_vals = sorted(values)
    idx = min(int(len(sorted_vals) * pct / 100.0), len(sorted_vals) - 1)
    return sorted_vals[idx]


def multiple_testing_corrected_percentile(seq_length: int, motif_length: int,
                                           target_expected_fp: float = 1.0, both_strands: bool = True) -> float:
    """A flat percentile (e.g. 99) means roughly (100-99)% of ALL windows
    tested will clear the bar by pure chance. With a 1200bp sequence and a
    15bp motif, that's ~2400 windows -- at p99 you'd expect ~24 "hits" in
    a sequence with NO real motif at all. This computes the percentile
    needed to keep the expected chance-hit count per sequence near
    target_expected_fp instead, so the same statistical strictness holds
    regardless of how long your sequences are."""
    n_windows = (seq_length - motif_length + 1) * (2 if both_strands else 1)
    n_windows = max(n_windows, 1)
    return 100 * (1 - target_expected_fp / n_windows)


def calibrate(pwms: dict, background_seqs: list, pct: Optional[float] = None,
              seq_length_for_correction: Optional[int] = None, target_expected_fp: float = 1.0) -> dict:
    thresholds = {}
    for name, pwm in pwms.items():
        motif_pct = pct if pct is not None else 99.0
        if seq_length_for_correction is not None:
            motif_pct = multiple_testing_corrected_percentile(
                seq_length_for_correction, len(pwm.matrix), target_expected_fp)
        all_scores = []
        for seq in background_seqs:
            all_scores.extend(score_all_windows(seq, pwm))
        threshold = percentile(all_scores, motif_pct)
        thresholds[name] = round(threshold, 4)
        print(f"{name:12s} background n_windows={len(all_scores):6d}  "
              f"percentile={motif_pct:.3f}  threshold={threshold:.3f}")
    return thresholds


def main(pwm_path: str, reference_path: str, percentile_val: float, output_path: str,
         n_random: int = 100, random_length: int = 500,
         auto_correct_for_length: int | None = None, target_expected_fp: float = 1.0):
    pwms = load_jaspar_pfms(pwm_path)
    print(f"Loaded {len(pwms)} motifs: {list(pwms.keys())}\n")

    background = random_background(n_random, random_length)

    if reference_path:
        with open(reference_path) as f:
            real_seqs = [row["sequence"] for row in csv.DictReader(f)]
        shuffled = [dinucleotide_shuffle(s, seed=i) for i, s in enumerate(real_seqs)]
        background.extend(shuffled)
        print(f"Background: {n_random} random + {len(shuffled)} dinucleotide-shuffled real sequences\n")
    else:
        print(f"Background: {n_random} random sequences only (no --reference given)\n")

    if auto_correct_for_length:
        print(f"Using multiple-testing-corrected percentile, targeting "
              f"~{target_expected_fp} expected false positive per {auto_correct_for_length}bp sequence\n")
    thresholds = calibrate(pwms, background, percentile_val,
                            seq_length_for_correction=auto_correct_for_length,
                            target_expected_fp=target_expected_fp)

    with open(output_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nWrote per-motif thresholds to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pwms", required=True)
    parser.add_argument("--reference", default="", help="CSV of real sequences to shuffle as extra background")
    parser.add_argument("--percentile", type=float, default=99.0,
                         help="Used only if --auto-correct-length is not given")
    parser.add_argument("--auto-correct-length", type=int, default=None,
                         help="Sequence length to correct for (e.g. 1201) -- recommended over --percentile "
                              "when sequences are long enough that many windows are tested per sequence")
    parser.add_argument("--target-fp", type=float, default=1.0,
                         help="Target expected false positives per sequence when using --auto-correct-length")
    parser.add_argument("--output", default="data/motif_thresholds.json")
    parser.add_argument("--n-random", type=int, default=100)
    args = parser.parse_args()
    main(args.pwms, args.reference, args.percentile, args.output, n_random=args.n_random,
         auto_correct_for_length=args.auto_correct_length, target_expected_fp=args.target_fp)
