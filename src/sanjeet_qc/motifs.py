"""PWM motif scanning on both DNA strands."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .models import BASES, MotifHit, MotifSpec


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def load_motifs(path: str | Path) -> list[MotifSpec]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        payload = payload.get("motifs", payload)
        if isinstance(payload, dict):
            payload = [dict(value, id=key) for key, value in payload.items()]
    if not isinstance(payload, list):
        raise ValueError("Motif JSON must contain a list or a 'motifs' list")
    motifs: list[MotifSpec] = []
    for item in payload:
        if not isinstance(item, dict) or "id" not in item or "matrix" not in item:
            raise ValueError("Each motif requires id and matrix")
        motifs.append(MotifSpec(
            id=str(item["id"]),
            name=item.get("name"),
            matrix=tuple(tuple(float(value) for value in row) for row in item["matrix"]),
            matrix_type=item.get("matrix_type", "probabilities"),
            threshold=float(item.get("threshold", 0.80)),
            role=item.get("role", "target"),
            conditions=tuple(str(value) for value in item.get("conditions", [])),
            unwanted_conditions=tuple(str(value) for value in item.get("unwanted_conditions", [])),
            pseudocount=float(item.get("pseudocount", 0.1)),
            metadata=dict(item.get("metadata", {})),
        ))
    return motifs


def scan_motifs(
    sequence: str,
    motifs: list[MotifSpec],
    step: int = 1,
    condition: str | None = None,
) -> list[MotifHit]:
    sequence = sequence.upper()
    hits: list[MotifHit] = []
    for motif in motifs:
        effective_role = motif.role_for_condition(condition)
        probabilities = _probability_matrix(motif)
        width = len(probabilities)
        for start in range(0, max(0, len(sequence) - width + 1), max(1, step)):
            window = sequence[start:start + width]
            if len(window) != width or not set(window) <= set(BASES):
                continue
            for strand, oriented in (("+", window), ("-", reverse_complement(window))):
                score = _normalized_score(oriented, probabilities)
                if score >= motif.threshold:
                    hits.append(MotifHit(
                        motif_id=motif.id,
                        motif_name=motif.name or motif.id,
                        start=start,
                        end=start + width,
                        strand=strand,
                        score=round(score, 6),
                        matched_sequence=oriented,
                        role=effective_role,
                    ))
    return hits


def motif_summary(hits: list[MotifHit]) -> dict[str, Any]:
    by_id: dict[str, int] = {}
    best_by_id: dict[str, float] = {}
    for hit in hits:
        by_id[hit.motif_id] = by_id.get(hit.motif_id, 0) + 1
        best_by_id[hit.motif_id] = max(best_by_id.get(hit.motif_id, 0.0), hit.score)
    return {
        "total_hits": len(hits),
        "target_hit_count": sum(hit.role == "target" for hit in hits),
        "unwanted_hit_count": sum(hit.role == "unwanted" for hit in hits),
        "neutral_hit_count": sum(hit.role == "neutral" for hit in hits),
        "hits_by_motif": by_id,
        "best_score_by_motif": best_by_id,
    }


def _probability_matrix(motif: MotifSpec) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    for row in motif.matrix:
        values = list(row)
        if motif.matrix_type == "counts":
            values = [value + motif.pseudocount for value in values]
        if any(value < 0 for value in values) or sum(values) <= 0:
            raise ValueError(f"Motif {motif.id!r} contains an invalid matrix row")
        total = sum(values)
        rows.append(tuple(value / total for value in values))
    return tuple(rows)


def _normalized_score(window: str, probabilities: tuple[tuple[float, ...], ...]) -> float:
    background = 0.25
    score = min_score = max_score = 0.0
    for base, row in zip(window, probabilities):
        index = BASES.index(base)
        values = [math.log2(max(value, 1e-12) / background) for value in row]
        score += values[index]
        min_score += min(values)
        max_score += max(values)
    if max_score == min_score:
        return 0.0
    return max(0.0, min(1.0, (score - min_score) / (max_score - min_score)))
