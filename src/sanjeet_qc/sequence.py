"""Sequence-level QC checks."""

from __future__ import annotations

import math
from collections import Counter

from .models import Issue, QCConfig


def normalize_sequence(sequence: str) -> str:
    return "".join(sequence.split()).upper()


def sequence_checks(sequence: str, config: QCConfig) -> tuple[dict[str, float | int | str], list[Issue]]:
    sequence = normalize_sequence(sequence)
    issues: list[Issue] = []
    invalid = sorted(set(sequence) - set("ACGT"))
    if invalid:
        issues.append(Issue(
            "invalid_alphabet",
            "error" if config.error_on_invalid_alphabet else "warning",
            f"Sequence contains invalid DNA symbols: {', '.join(invalid)}.",
            {"invalid_symbols": invalid},
        ))

    length = len(sequence)
    if not config.min_length <= length <= config.max_length:
        issues.append(Issue(
            "invalid_length",
            "error" if config.error_on_length else "warning",
            f"Length {length} is outside the allowed range {config.min_length}-{config.max_length} bp.",
            {"length": length, "min_length": config.min_length, "max_length": config.max_length},
        ))

    valid_bases = [base for base in sequence if base in "ACGT"]
    gc = ((valid_bases.count("G") + valid_bases.count("C")) / len(valid_bases)) if valid_bases else 0.0
    if valid_bases and not config.min_gc <= gc <= config.max_gc:
        issues.append(Issue(
            "gc_out_of_range",
            "warning",
            f"GC content {gc:.3f} is outside the preferred range {config.min_gc:.3f}-{config.max_gc:.3f}.",
            {"gc_content": gc, "min_gc": config.min_gc, "max_gc": config.max_gc},
        ))

    max_run, run_base = longest_homopolymer(sequence)
    if max_run > config.max_homopolymer:
        issues.append(Issue(
            "long_homopolymer",
            "warning",
            f"Longest homopolymer is {run_base * max_run} ({max_run} bases), above the limit of {config.max_homopolymer}.",
            {"run_length": max_run, "base": run_base},
        ))

    repeats = repeated_kmers(sequence, config.repeat_k)
    repeat_fraction = repeated_base_fraction(sequence, repeats, config.repeat_k)
    if repeats and max(repeats.values()) > config.max_repeat_count:
        worst_kmer, count = max(repeats.items(), key=lambda item: item[1])
        issues.append(Issue(
            "excessive_repeat",
            "warning",
            f"The {config.repeat_k}-mer {worst_kmer} occurs {count} times, above the limit of {config.max_repeat_count}.",
            {"kmer": worst_kmer, "count": count, "repeat_k": config.repeat_k},
        ))
    if repeat_fraction > config.max_repeat_fraction:
        issues.append(Issue(
            "high_repeat_fraction",
            "warning",
            f"Repeated k-mers cover approximately {repeat_fraction:.3f} of the sequence.",
            {"repeat_fraction": repeat_fraction, "max_repeat_fraction": config.max_repeat_fraction},
        ))

    complexity = normalized_kmer_entropy(sequence, config.complexity_k)
    if complexity < config.min_complexity:
        issues.append(Issue(
            "low_complexity",
            "warning",
            f"Normalized {config.complexity_k}-mer entropy is {complexity:.3f}, below {config.min_complexity:.3f}.",
            {"complexity": complexity, "complexity_k": config.complexity_k},
        ))

    metrics = {
        "length": length,
        "gc_content": round(gc, 6),
        "longest_homopolymer": max_run,
        "homopolymer_base": run_base,
        "repeat_k": config.repeat_k,
        "repeated_kmer_count": len(repeats),
        "max_kmer_occurrence": max(repeats.values(), default=0),
        "repeat_fraction": round(repeat_fraction, 6),
        "complexity": round(complexity, 6),
    }
    return metrics, issues


def longest_homopolymer(sequence: str) -> tuple[int, str]:
    if not sequence:
        return 0, ""
    best_length = current_length = 1
    best_base = current_base = sequence[0]
    for base in sequence[1:]:
        if base == current_base:
            current_length += 1
        else:
            if current_length > best_length:
                best_length, best_base = current_length, current_base
            current_base, current_length = base, 1
    if current_length > best_length:
        best_length, best_base = current_length, current_base
    return best_length, best_base


def repeated_kmers(sequence: str, k: int) -> dict[str, int]:
    if k <= 0 or len(sequence) < k:
        return {}
    counts = Counter(sequence[index:index + k] for index in range(len(sequence) - k + 1))
    return {kmer: count for kmer, count in counts.items() if count > 1}


def repeated_base_fraction(sequence: str, repeats: dict[str, int], k: int) -> float:
    if not sequence or not repeats or k <= 0:
        return 0.0
    covered: set[int] = set()
    for kmer in repeats:
        start = sequence.find(kmer)
        while start >= 0:
            covered.update(range(start, min(start + k, len(sequence))))
            start = sequence.find(kmer, start + 1)
    return len(covered) / len(sequence)


def normalized_kmer_entropy(sequence: str, k: int) -> float:
    if not sequence or k <= 0 or len(sequence) < k:
        return 0.0
    counts = Counter(sequence[index:index + k] for index in range(len(sequence) - k + 1))
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    maximum = math.log2(min(4 ** k, total))
    return entropy / maximum if maximum else 0.0
