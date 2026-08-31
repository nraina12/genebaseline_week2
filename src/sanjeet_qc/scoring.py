"""Transparent Candidate Regulatory Sequence Score."""

from __future__ import annotations

from collections import defaultdict

from .models import CandidateResult, MotifSpec, QCConfig


def score_candidate(
    result: CandidateResult,
    motifs: list[MotifSpec],
    config: QCConfig,
    training_similarity: float | None,
    diversity: float,
) -> None:
    condition_value = result.metadata.get("condition")
    condition = str(condition_value).strip() if condition_value is not None else None
    condition = condition or None
    all_targets = [motif for motif in motifs if motif.role == "target"]
    targets = [motif for motif in motifs if motif.role_for_condition(condition) == "target"]
    target_hits = [hit for hit in result.motif_hits if hit.role == "target"]
    unwanted_hits = [hit for hit in result.motif_hits if hit.role == "unwanted"]

    best_by_motif: dict[str, float] = defaultdict(float)
    for hit in target_hits:
        best_by_motif[hit.motif_id] = max(best_by_motif[hit.motif_id], hit.score)
    if targets:
        strength = sum(best_by_motif.values()) / len(targets)
        count = min(1.0, len(best_by_motif) / len(targets))
    elif all_targets:
        strength = 0.0
        count = 0.0
    else:
        strength = 1.0
        count = 1.0
    complexity = float(result.metrics.get("complexity", 0.0))
    gc = float(result.metrics.get("gc_content", 0.0))
    composition = _composition_score(gc, config.min_gc, config.max_gc)
    training_component = 0.5 if training_similarity is None else training_similarity
    if training_similarity is None:
        training_text = "No training set supplied; training similarity received a neutral value of 0.500."
    else:
        training_text = f"Nearest training-sequence similarity was {training_similarity:.3f}."

    components = {
        "target_motif_strength": _bounded(strength),
        "appropriate_motif_count": _bounded(count),
        "sequence_complexity": _bounded(complexity),
        "diversity": _bounded(diversity),
        "composition": _bounded(composition),
        "training_similarity": _bounded(training_component),
    }
    total_weight = sum(config.score_weights.values()) or 1.0
    score = 100 * sum(components[key] * config.score_weights.get(key, 0.0) for key in components) / total_weight
    explanations = [
        f"Target motif strength: {components['target_motif_strength']:.3f}.",
        f"Target motif coverage: {len(best_by_motif)}/{len(targets) or 0} ({components['appropriate_motif_count']:.3f}).",
        f"Sequence complexity: {components['sequence_complexity']:.3f}.",
        f"Batch diversity: {components['diversity']:.3f}.",
        f"Composition suitability: {components['composition']:.3f} with GC={gc:.3f}.",
        training_text,
    ]
    if condition:
        explanations.insert(0, f"Condition-specific motif interpretation used label {condition!r}.")
    elif any(motif.conditions or motif.unwanted_conditions for motif in motifs):
        explanations.insert(0, "No condition label was supplied; condition-specific motifs were treated as neutral.")
    penalty = 0.0
    if unwanted_hits:
        penalty += min(40.0, 10.0 * len(unwanted_hits))
        explanations.append(f"Penalty: {len(unwanted_hits)} unwanted motif hit(s).")
    errors = sum(issue.severity == "error" for issue in result.issues)
    warnings = sum(issue.severity == "warning" for issue in result.issues)
    if errors:
        penalty += min(50.0, 20.0 * errors)
        explanations.append(f"Penalty: {errors} blocking error(s).")
    if warnings:
        penalty += min(20.0, 3.0 * warnings)
        explanations.append(f"Penalty: {warnings} warning(s).")

    result.score_components = {key: round(value, 6) for key, value in components.items()}
    result.score = round(max(0.0, min(100.0, score - penalty)), 3)
    result.score_explanation = explanations


def _composition_score(gc: float, minimum: float, maximum: float) -> float:
    if minimum <= gc <= maximum:
        return 1.0
    if gc < minimum:
        return max(0.0, gc / minimum) if minimum else 0.0
    return max(0.0, (1 - gc) / (1 - maximum)) if maximum < 1 else 0.0


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, value))
