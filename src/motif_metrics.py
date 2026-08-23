"""
Entrypoint:
    python src/motif_metrics.py --input outputs/generated_XXXX.jsonl \
        --pwms data/jaspar_pwms.txt --threshold 0.8
"""

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

BASE_ORDER = ["A", "C", "G", "T"]
BASE_TO_IDX = {b: i for i, b in enumerate(BASE_ORDER)}
COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}

@dataclass
class PWM:
    name: str
    matrix: List[List[float]]  # log-odds scores, shape (length, 4), columns A,C,G,T
    max_score: float
    min_score: float


def _pfm_to_pwm(counts: List[List[float]], pseudocount: float = 0.8, background=(0.25, 0.25, 0.25, 0.25)) -> List[List[float]]:
    """counts: list of 4 rows (A,C,G,T) each length L -> transpose to
    per-position log-odds scores relative to background."""
    length = len(counts[0])
    pwm = []
    for pos in range(length):
        col_counts = [counts[b][pos] for b in range(4)]
        total = sum(col_counts) + 4 * pseudocount
        row = []
        for b in range(4):
            freq = (col_counts[b] + pseudocount) / total
            row.append(math.log2(freq / background[b]))
        pwm.append(row)
    return pwm


def load_jaspar_pfms(path: str) -> Dict[str, PWM]:
    motifs: Dict[str, PWM] = {}
    with open(path) as f:
        text = f.read()

    records = re.split(r"(?=^>)", text, flags=re.MULTILINE)
    for record in records:
        record = record.strip()
        if not record:
            continue
        lines = record.splitlines()
        header = lines[0].lstrip(">").strip()
        name = header.split()[0] if header else "UNKNOWN"

        counts = []
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            # strip leading base letter and brackets: "A [ 1 2 8 0 ]" -> "1 2 8 0"
            nums = re.findall(r"-?\d+\.?\d*", line)
            counts.append([float(n) for n in nums])
        if len(counts) != 4:
            continue  # malformed record, skip

        pwm_matrix = _pfm_to_pwm(counts)
        scores = [max(row) for row in pwm_matrix]
        min_scores = [min(row) for row in pwm_matrix]
        motifs[name] = PWM(
            name=name,
            matrix=pwm_matrix,
            max_score=sum(scores),
            min_score=sum(min_scores),
        )
    return motifs


@dataclass
class MotifMatch:
    motif_name: str
    position: int       # 0-indexed start position on the + strand coordinate frame
    strand: str          # "+" or "-"
    score: float          # raw log-odds sum
    normalized_score: float  # 0-1, (score - min) / (max - min)


def reverse_complement(seq: str) -> str:
    return "".join(COMPLEMENT.get(b, "N") for b in reversed(seq.upper()))


def _score_window(window: str, pwm: PWM) -> float:
    score = 0.0
    for i, base in enumerate(window):
        idx = BASE_TO_IDX.get(base)
        if idx is None:
            return float("-inf")  # ambiguous base (N, padding) -> reject window
        score += pwm.matrix[i][idx]
    return score


def scan_sequence(seq: str, pwm: PWM, threshold: float = 0.8, both_strands: bool = True) -> List[MotifMatch]:
    """threshold is a normalized-score cutoff in [0,1] relative to the PWM's
    own min/max possible score, so it's comparable across different motifs
    of different lengths/information content."""
    seq = seq.upper()
    length = len(pwm.matrix)
    matches = []
    score_range = max(pwm.max_score - pwm.min_score, 1e-8)

    strands = [("+", seq)]
    if both_strands:
        strands.append(("-", reverse_complement(seq)))

    for strand, s in strands:
        for start in range(0, len(s) - length + 1):
            window = s[start:start + length]
            raw = _score_window(window, pwm)
            if raw == float("-inf"):
                continue
            norm = (raw - pwm.min_score) / score_range
            if norm >= threshold:
                pos = start if strand == "+" else len(seq) - start - length
                matches.append(MotifMatch(pwm.name, pos, strand, raw, norm))
    return matches


def motif_frequency(matches: List[MotifMatch], seq_length: int) -> float:
    """Matches per 100bp, per the framework's stated normalization."""
    return 100.0 * len(matches) / max(seq_length, 1)


def motif_strength_stats(matches: List[MotifMatch]) -> dict:
    if not matches:
        return {"mean_normalized_score": None, "max_normalized_score": None, "n_matches": 0}
    scores = [m.normalized_score for m in matches]
    return {
        "mean_normalized_score": sum(scores) / len(scores),
        "max_normalized_score": max(scores),
        "n_matches": len(matches),
    }


def motif_arrangement(matches: List[MotifMatch], seq_length: int, cluster_window: int = 30) -> dict:
    if len(matches) < 2:
        return {
            "positions": [m.position for m in matches],
            "pairwise_spacing": [],
            "order_by_position": [m.motif_name for m in sorted(matches, key=lambda m: m.position)],
            "n_clusters": 1 if matches else 0,
        }

    sorted_matches = sorted(matches, key=lambda m: m.position)
    positions = [m.position for m in sorted_matches]
    spacing = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]

    # simple clustering: group matches whose gap to the next is <= cluster_window
    clusters = 1
    for gap in spacing:
        if gap > cluster_window:
            clusters += 1

    return {
        "positions": positions,
        "pairwise_spacing": spacing,
        "order_by_position": [m.motif_name for m in sorted_matches],
        "n_clusters": clusters,
    }


def summarize_sequence(seq: str, condition: str, pwms: Dict[str, PWM], threshold: float,
                        relevant_motifs_by_condition: Optional[Dict[str, List[str]]] = None) -> dict:
    all_matches: Dict[str, List[MotifMatch]] = {}
    for name, pwm in pwms.items():
        all_matches[name] = scan_sequence(seq, pwm, threshold=threshold)

    relevant = (relevant_motifs_by_condition or {}).get(condition, list(pwms.keys()))
    relevant_matches = [m for name in relevant for m in all_matches.get(name, [])]

    return {
        "condition": condition,
        "sequence_length": len(seq),
        "relevant_motif_frequency_per_100bp": motif_frequency(relevant_matches, len(seq)),
        "relevant_motif_strength": motif_strength_stats(relevant_matches),
        "relevant_motif_arrangement": motif_arrangement(relevant_matches, len(seq)),
        "per_motif_match_counts": {name: len(m) for name, m in all_matches.items()},
    }


def main(input_path: str, pwm_path: str, threshold: float):
    pwms = load_jaspar_pfms(pwm_path)
    print(f"Loaded {len(pwms)} PWMs: {list(pwms.keys())}")

    results = []
    with open(input_path) as f:
        for line in f:
            rec = json.loads(line)
            seq = rec["sequence"]
            condition = rec.get("condition") or "unknown"
            results.append(summarize_sequence(seq, condition, pwms, threshold))

    out_path = input_path.rsplit(".", 1)[0] + "_motif_metrics.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote motif metrics for {len(results)} sequences to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help=".jsonl file with a 'sequence' and 'condition' field per line")
    parser.add_argument("--pwms", required=True, help="Path to JASPAR PFM text file (from Tisha's annotation pull)")
    parser.add_argument("--threshold", type=float, default=0.8,
                         help="Normalized motif-match score cutoff, 0-1 (default 0.8)")
    args = parser.parse_args()
    main(args.input, args.pwms, args.threshold)
