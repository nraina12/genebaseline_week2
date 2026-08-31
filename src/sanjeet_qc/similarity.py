"""Near-duplicate and diversity heuristics."""

from __future__ import annotations

from .models import DuplicateMatch


def sequence_similarity(a: str, b: str, k: int = 8) -> tuple[float, str]:
    a, b = a.upper(), b.upper()
    if len(a) == len(b):
        if not a:
            return 1.0, "hamming_identity"
        return sum(x == y for x, y in zip(a, b)) / len(a), "hamming_identity"
    left, right = set(_kmers(a, k)), set(_kmers(b, k))
    if not left and not right:
        return (1.0 if a == b else 0.0), "kmer_jaccard"
    return len(left & right) / len(left | right), "kmer_jaccard"


def find_batch_duplicates(records: list[tuple[str, str]], threshold: float, k: int) -> dict[str, list[DuplicateMatch]]:
    matches: dict[str, list[DuplicateMatch]] = {record_id: [] for record_id, _ in records}
    for index, (record_id, sequence) in enumerate(records):
        for other_id, other_sequence in records[index + 1:]:
            if record_id == other_id:
                continue
            similarity, method = sequence_similarity(sequence, other_sequence, k)
            if similarity >= threshold:
                match = DuplicateMatch(other_id, round(similarity, 6), method, "batch")
                reverse = DuplicateMatch(record_id, round(similarity, 6), method, "batch")
                matches[record_id].append(match)
                matches[other_id].append(reverse)
    return matches


def find_training_duplicates(
    candidates: list[tuple[str, str]], training: list[tuple[str, str]], threshold: float, k: int
) -> dict[str, list[DuplicateMatch]]:
    matches: dict[str, list[DuplicateMatch]] = {candidate_id: [] for candidate_id, _ in candidates}
    for candidate_id, sequence in candidates:
        for training_id, training_sequence in training:
            similarity, method = sequence_similarity(sequence, training_sequence, k)
            if similarity >= threshold:
                matches[candidate_id].append(
                    DuplicateMatch(training_id, round(similarity, 6), method, "training")
                )
    return matches


def max_similarity(sequence: str, references: list[str], k: int = 8) -> float | None:
    if not references:
        return None
    return max(sequence_similarity(sequence, reference, k)[0] for reference in references)


def _kmers(sequence: str, k: int) -> list[str]:
    if k <= 0 or len(sequence) < k:
        return []
    return [sequence[index:index + k] for index in range(len(sequence) - k + 1)]
