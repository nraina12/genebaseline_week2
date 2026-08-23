"""Candidate Regulatory Sequence Score (CRSS), per Sanjeet's spec.

Deliberately NOT a single opaque "Promoter Quality Score" -- every
candidate gets a full breakdown of which components contributed what, and
in the failing case, exactly which check(s) failed and why. That
transparency is the point of the spec, not an afterthought.

Checks / components:
  1. length_alphabet   -- valid bases only, length in expected range
  2. gc_content         -- composition sanity range
  3. homopolymers        -- long single-base runs (e.g. AAAAAAAA) flagged
  4. tandem_repeats       -- short k-mer excessively repeated in a row
  5. near_duplicate        -- max similarity to OTHER candidates in this batch
  6. similarity_to_training -- max similarity to real training/reference
                                 sequences (overfitting/memorization check)
  7. target_motif_strength  -- PWM score for condition-relevant motifs
  8. target_motif_count      -- number of relevant motif matches (too few
                                  or implausibly many are both penalized)
  9. unwanted_motif_penalty   -- matches for OTHER conditions' motifs
                                   (esp. PTF1A appearing in tumor sequences)
 10. complexity               -- unique k-mer fraction (catches degenerate/
                                   low-information sequences a naive model
                                   might produce, e.g. near-constant output)

Entrypoint:
    python src/candidate_score.py --input outputs/generated_XXXX.jsonl \
        --pwms data/jaspar_pwms.txt --reference data/real_sequences.csv
"""

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from motif_metrics import load_jaspar_pfms, scan_sequence, PWM

VALID_BASES = set("ACGT")

# which motifs count as "target" vs "unwanted" per condition -- keep in
# sync with data/jaspar_pwms.txt and compare_to_real.py's mapping
RELEVANT_MOTIFS_BY_CONDITION = {
    "KRAS_MAPK_ERK": ["MA0028.2", "MA0099.3"],
    "HNF4G_FOXA1": ["MA0484.1", "MA0148.4"],
    "GATA6": ["MA1104.1"],
    "PTF1A_NEGATIVE": ["MA1619.1"],
}

# default component weights -- transparent and adjustable, not baked in.
# Team should tune these together rather than treat them as fixed.
DEFAULT_WEIGHTS = {
    "target_motif_strength": 0.25,
    "target_motif_count": 0.15,
    "complexity": 0.15,
    "composition": 0.10,
    "diversity": 0.10,
    "similarity_to_training": 0.10,   # penalty component (inverted below)
    "unwanted_motif_penalty": 0.15,   # penalty component (inverted below)
}


@dataclass
class CheckResult:
    name: str
    passed: bool
    value: Optional[float]
    detail: str


@dataclass
class CandidateScore:
    sequence_id: int
    condition: str
    overall_score: float          # 0-100
    passed: bool                   # did it clear the hard gates (checks 1-4)?
    component_scores: Dict[str, float] = field(default_factory=dict)
    checks: List[CheckResult] = field(default_factory=list)
    explanation: str = ""


# ---------------------------------------------------------------------------
# Hard gate checks (1-4): binary pass/fail, independent of the weighted score
# ---------------------------------------------------------------------------

def check_length_alphabet(seq: str, min_len: int = 200, max_len: int = 500) -> CheckResult:
    invalid = set(seq.upper()) - VALID_BASES
    length_ok = min_len <= len(seq) <= max_len
    passed = (not invalid) and length_ok
    detail = []
    if invalid:
        detail.append(f"invalid characters: {sorted(invalid)}")
    if not length_ok:
        detail.append(f"length {len(seq)} outside [{min_len},{max_len}]")
    return CheckResult("length_alphabet", passed, len(seq), "; ".join(detail) or "OK")


def gc_content(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / max(len(seq), 1)


def check_gc_content(seq: str, min_gc: float = 0.25, max_gc: float = 0.75) -> CheckResult:
    gc = gc_content(seq)
    passed = min_gc <= gc <= max_gc
    return CheckResult("gc_content", passed, round(gc, 3),
                        "OK" if passed else f"GC {gc:.2f} outside [{min_gc},{max_gc}]")


def check_homopolymers(seq: str, max_run: int = 8) -> CheckResult:
    longest = 1
    current = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    passed = longest <= max_run
    return CheckResult("homopolymers", passed, longest,
                        "OK" if passed else f"longest run {longest}bp exceeds max {max_run}bp")


def check_tandem_repeats(seq: str, k_range=(2, 6), min_repeats: int = 5) -> CheckResult:
    """Flags a k-mer that repeats consecutively min_repeats+ times, e.g.
    'ATATATATATAT' (k=2, 6 repeats)."""
    worst_repeats = 0
    worst_kmer = None
    for k in range(k_range[0], k_range[1] + 1):
        i = 0
        while i < len(seq) - k:
            kmer = seq[i:i + k]
            repeats = 1
            j = i + k
            while seq[j:j + k] == kmer and j + k <= len(seq):
                repeats += 1
                j += k
            if repeats > worst_repeats:
                worst_repeats = repeats
                worst_kmer = kmer
            i = j if repeats > 1 else i + 1
    passed = worst_repeats < min_repeats
    detail = "OK" if passed else f"'{worst_kmer}' repeats {worst_repeats}x consecutively"
    return CheckResult("tandem_repeats", passed, worst_repeats, detail)


# ---------------------------------------------------------------------------
# Similarity components (5-6)
# ---------------------------------------------------------------------------

def hamming_similarity(a: str, b: str) -> float:
    length = min(len(a), len(b))
    if length == 0:
        return 0.0
    matches = sum(1 for i in range(length) if a[i] == b[i])
    return matches / length


def max_similarity(seq: str, others: List[str]) -> float:
    if not others:
        return 0.0
    return max(hamming_similarity(seq, o) for o in others)


# ---------------------------------------------------------------------------
# Complexity (10)
# ---------------------------------------------------------------------------

def sequence_complexity(seq: str, k: int = 4) -> float:
    """Fraction of possible k-mers actually observed, relative to what's
    achievable given the sequence length -- low values flag degenerate/
    repetitive/low-information output."""
    if len(seq) < k:
        return 0.0
    kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
    unique = len(set(kmers))
    max_possible = min(4 ** k, len(kmers))
    return unique / max_possible if max_possible > 0 else 0.0


# ---------------------------------------------------------------------------
# Motif components (7-9)
# ---------------------------------------------------------------------------

def motif_components(seq: str, condition: str, pwms: Dict[str, PWM], threshold=0.75) -> dict:
    """threshold: single float or {motif_name: threshold} dict from
    calibrate_motif_thresholds.py (recommended)."""
    relevant_names = RELEVANT_MOTIFS_BY_CONDITION.get(condition, [])
    unwanted_names = [n for cond, names in RELEVANT_MOTIFS_BY_CONDITION.items()
                       if cond != condition for n in names]

    def t(name):
        return threshold[name] if isinstance(threshold, dict) else threshold

    relevant_matches = []
    for name in relevant_names:
        if name in pwms:
            relevant_matches.extend(scan_sequence(seq, pwms[name], threshold=t(name)))

    unwanted_matches = []
    for name in unwanted_names:
        if name in pwms:
            unwanted_matches.extend(scan_sequence(seq, pwms[name], threshold=t(name)))

    strength = (sum(m.normalized_score for m in relevant_matches) / len(relevant_matches)
                if relevant_matches else 0.0)
    return {
        "relevant_matches": relevant_matches,
        "unwanted_matches": unwanted_matches,
        "strength": strength,
        "count": len(relevant_matches),
    }


# ---------------------------------------------------------------------------
# Overall scoring
# ---------------------------------------------------------------------------

def score_candidate(seq: str, condition: str, sequence_id: int, pwms: Dict[str, PWM],
                     batch_sequences: Optional[List[str]] = None,
                     training_sequences: Optional[List[str]] = None,
                     weights: Optional[dict] = None, threshold: float = 0.75,
                     target_motif_count_range=(1, 8)) -> CandidateScore:
    weights = weights or DEFAULT_WEIGHTS
    checks = [
        check_length_alphabet(seq),
        check_gc_content(seq),
        check_homopolymers(seq),
        check_tandem_repeats(seq),
    ]
    hard_gate_passed = all(c.passed for c in checks)

    motifs = motif_components(seq, condition, pwms, threshold)
    complexity = sequence_complexity(seq)
    gc = gc_content(seq)
    composition_score = 1.0 - abs(gc - 0.5) * 2  # peaks at GC=0.5, 0 at extremes -- simple, adjustable

    others = [s for s in (batch_sequences or []) if s != seq]
    diversity_score = 1.0 - max_similarity(seq, others)  # higher = more distinct from batch-mates
    train_similarity = max_similarity(seq, training_sequences or [])

    # motif count score: penalize both too few AND implausibly many (the
    # latter suggesting the model is just stuffing motifs rather than
    # producing realistic regulatory architecture)
    lo, hi = target_motif_count_range
    if motifs["count"] < lo:
        count_score = motifs["count"] / lo
    elif motifs["count"] > hi:
        count_score = max(0.0, 1.0 - (motifs["count"] - hi) / hi)
    else:
        count_score = 1.0

    unwanted_penalty = min(1.0, len(motifs["unwanted_matches"]) / 5.0)  # scales 0-1, caps at 5+ matches

    component_scores = {
        "target_motif_strength": motifs["strength"],
        "target_motif_count": count_score,
        "complexity": complexity,
        "composition": composition_score,
        "diversity": diversity_score,
        "similarity_to_training": 1.0 - train_similarity,   # inverted: LOWER similarity scores higher
        "unwanted_motif_penalty": 1.0 - unwanted_penalty,     # inverted: fewer unwanted matches scores higher
    }

    overall = 100.0 * sum(weights[k] * component_scores[k] for k in weights)

    explanation_parts = [f"{c.name}: {'PASS' if c.passed else 'FAIL'} ({c.detail})" for c in checks]
    explanation_parts.append(
        f"target motifs: {motifs['count']} matches, mean strength {motifs['strength']:.2f}"
    )
    if motifs["unwanted_matches"]:
        explanation_parts.append(f"WARNING: {len(motifs['unwanted_matches'])} unwanted motif match(es) found")
    explanation_parts.append(f"complexity {complexity:.2f}, GC {gc:.2f}, "
                              f"max similarity to training set {train_similarity:.2f}")

    return CandidateScore(
        sequence_id=sequence_id,
        condition=condition,
        overall_score=round(overall, 2),
        passed=hard_gate_passed,
        component_scores={k: round(v, 3) for k, v in component_scores.items()},
        checks=checks,
        explanation=" | ".join(explanation_parts),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_generated(path: str) -> List[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_reference_sequences(path: str) -> List[str]:
    seqs = []
    with open(path) as f:
        for row in csv.DictReader(f):
            seqs.append(row["sequence"])
    return seqs


def main(input_path: str, pwm_path: str, reference_path: str, threshold, thresholds_path: str = ""):
    pwms = load_jaspar_pfms(pwm_path)
    if thresholds_path:
        with open(thresholds_path) as f:
            threshold = json.load(f)
        print(f"Using calibrated per-motif thresholds from {thresholds_path}")
    records = load_generated(input_path)
    training_seqs = load_reference_sequences(reference_path) if reference_path else []
    batch_seqs = [r["sequence"] for r in records]

    results = []
    for i, rec in enumerate(records):
        score = score_candidate(
            rec["sequence"], rec.get("condition") or "unknown", i, pwms,
            batch_sequences=batch_seqs, training_sequences=training_seqs, threshold=threshold,
        )
        results.append(score)

    out_path = input_path.rsplit(".", 1)[0] + "_crss.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps({
                "sequence_id": r.sequence_id,
                "condition": r.condition,
                "overall_score": r.overall_score,
                "passed": r.passed,
                "component_scores": r.component_scores,
                "explanation": r.explanation,
            }) + "\n")

    n_passed = sum(1 for r in results if r.passed)
    print(f"Scored {len(results)} candidates -- {n_passed} passed hard gates.")
    print(f"Mean overall score: {sum(r.overall_score for r in results) / len(results):.1f}")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--pwms", required=True)
    parser.add_argument("--reference", default="", help="CSV of real/training sequences for similarity check")
    parser.add_argument("--threshold", type=float, default=0.75, help="Used if --thresholds not given")
    parser.add_argument("--thresholds", default="", help="Path to calibrated per-motif thresholds JSON "
                         "from calibrate_motif_thresholds.py (recommended over --threshold)")
    args = parser.parse_args()
    main(args.input, args.pwms, args.reference, args.threshold, args.thresholds)
