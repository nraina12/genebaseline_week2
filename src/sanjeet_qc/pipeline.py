"""Pipeline orchestration."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .models import CandidateResult, Issue, MotifSpec, QCConfig, SequenceRecord
from .motifs import motif_summary, scan_motifs
from .scoring import score_candidate
from .sequence import normalize_sequence, sequence_checks
from .similarity import find_batch_duplicates, find_training_duplicates, max_similarity, sequence_similarity


def run_qc(
    records: list[SequenceRecord],
    motifs: list[MotifSpec] | None = None,
    training_records: list[SequenceRecord] | None = None,
    config: QCConfig | None = None,
) -> tuple[list[CandidateResult], dict[str, Any]]:
    config = config or QCConfig()
    motifs = motifs or []
    training_records = training_records or []
    duplicate_ids = {record_id for record_id in [record.id for record in records] if [record.id for record in records].count(record_id) > 1}
    candidate_pairs = [(record.id, normalize_sequence(record.sequence)) for record in records]
    training_pairs = [(record.id, normalize_sequence(record.sequence)) for record in training_records]
    batch_matches = find_batch_duplicates(candidate_pairs, config.duplicate_similarity, config.repeat_k)
    training_matches = find_training_duplicates(candidate_pairs, training_pairs, config.duplicate_similarity, config.repeat_k)
    training_sequences = [sequence for _, sequence in training_pairs]
    results: list[CandidateResult] = []

    for record, (record_id, sequence) in zip(records, candidate_pairs):
        metrics, issues = sequence_checks(sequence, config)
        if record_id in duplicate_ids:
            issues.append(Issue("duplicate_id", "error", f"Candidate ID {record_id!r} is not unique."))
        condition_value = record.metadata.get("condition")
        condition = str(condition_value).strip() if condition_value is not None else None
        condition = condition or None
        if any(motif.conditions or motif.unwanted_conditions for motif in motifs) and condition is None:
            issues.append(Issue(
                "missing_condition",
                "warning",
                "Candidate has no condition label, so condition-specific motif hits are treated as neutral.",
            ))
        hits = scan_motifs(sequence, motifs, config.motif_scan_step, condition)
        duplicates = batch_matches.get(record_id, []) + training_matches.get(record_id, [])
        if duplicates:
            severity = "error" if config.error_on_near_duplicate else "warning"
            issues.append(Issue(
                "near_duplicate",
                severity,
                f"Candidate is near-duplicate to {len(duplicates)} sequence(s).",
                {"matches": [asdict(match) for match in duplicates]},
            ))
        metrics["motifs"] = motif_summary(hits)
        metrics["batch_near_duplicate_count"] = len(batch_matches.get(record_id, []))
        metrics["training_near_duplicate_count"] = len(training_matches.get(record_id, []))
        references = [other_sequence for other_id, other_sequence in candidate_pairs if other_id != record_id]
        references.extend(training_sequences)
        similarities = [sequence_similarity(sequence, reference, config.repeat_k)[0] for reference in references]
        diversity = 1.0 - max(similarities) if similarities else 1.0
        result = CandidateResult(
            id=record_id,
            sequence=sequence,
            passed=not any(issue.severity == "error" for issue in issues),
            issues=issues,
            metrics=metrics,
            motif_hits=hits,
            duplicate_matches=duplicates,
            metadata=record.metadata,
        )
        score_candidate(result, motifs, config, max_similarity(sequence, training_sequences, config.repeat_k), diversity)
        results.append(result)

    summary = {
        "candidate_count": len(results),
        "passed_count": sum(result.passed for result in results),
        "failed_count": sum(not result.passed for result in results),
        "mean_score": round(sum(result.score for result in results) / len(results), 3) if results else 0.0,
        "motif_count": len(motifs),
        "config": asdict(config),
    }
    return results, summary
