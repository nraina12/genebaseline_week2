"""Typed data structures for the QC pipeline."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


BASES = "ACGT"
Severity = Literal["error", "warning", "info"]
MotifRole = Literal["target", "unwanted", "neutral"]


@dataclass(frozen=True)
class SequenceRecord:
    id: str
    sequence: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MotifSpec:
    """PWM rows in position order, with columns A/C/G/T."""

    id: str
    matrix: tuple[tuple[float, ...], ...]
    name: str | None = None
    matrix_type: Literal["probabilities", "counts"] = "probabilities"
    threshold: float = 0.80
    role: MotifRole = "target"
    conditions: tuple[str, ...] = ()
    unwanted_conditions: tuple[str, ...] = ()
    pseudocount: float = 0.1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.matrix or any(len(row) != 4 for row in self.matrix):
            raise ValueError(f"Motif {self.id!r} must have non-empty four-column rows")
        if self.matrix_type not in {"probabilities", "counts"}:
            raise ValueError("matrix_type must be probabilities or counts")
        values = [value for row in self.matrix for value in row]
        if any(not math.isfinite(value) for value in values):
            raise ValueError(f"Motif {self.id!r} matrix values must be finite")
        if any(value < 0 for value in values):
            raise ValueError(f"Motif {self.id!r} matrix values must be non-negative")
        if self.matrix_type == "probabilities" and any(sum(row) <= 0 for row in self.matrix):
            raise ValueError(f"Motif {self.id!r} probability rows must have a positive total")
        if not math.isfinite(self.pseudocount):
            raise ValueError(f"Motif {self.id!r} pseudocount must be finite")
        if self.pseudocount < 0:
            raise ValueError(f"Motif {self.id!r} pseudocount must be non-negative")
        if self.matrix_type == "counts" and any(
            sum(row) + 4 * self.pseudocount <= 0 for row in self.matrix
        ):
            raise ValueError(f"Motif {self.id!r} count rows must have a positive total")
        if not 0 <= self.threshold <= 1:
            raise ValueError(f"Motif {self.id!r} threshold must be between 0 and 1")
        if self.role not in {"target", "unwanted", "neutral"}:
            raise ValueError("role must be target, unwanted, or neutral")

    def role_for_condition(self, condition: str | None) -> MotifRole:
        """Resolve how a motif should be interpreted for one candidate label."""
        if condition and condition in self.unwanted_conditions:
            return "unwanted"
        if self.conditions:
            return self.role if condition and condition in self.conditions else "neutral"
        return self.role


@dataclass(frozen=True)
class Issue:
    code: str
    severity: Severity
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MotifHit:
    motif_id: str
    motif_name: str
    start: int
    end: int
    strand: Literal["+", "-"]
    score: float
    matched_sequence: str
    role: MotifRole


@dataclass(frozen=True)
class DuplicateMatch:
    other_id: str
    similarity: float
    method: Literal["hamming_identity", "kmer_jaccard"]
    source: Literal["batch", "training"]


@dataclass(frozen=True)
class QCConfig:
    min_length: int = 200
    max_length: int = 500
    min_gc: float = 0.30
    max_gc: float = 0.70
    max_homopolymer: int = 8
    repeat_k: int = 8
    max_repeat_count: int = 3
    max_repeat_fraction: float = 0.35
    complexity_k: int = 2
    min_complexity: float = 0.55
    duplicate_similarity: float = 0.95
    motif_scan_step: int = 1
    error_on_length: bool = True
    error_on_invalid_alphabet: bool = True
    error_on_near_duplicate: bool = True
    score_weights: dict[str, float] = field(default_factory=lambda: {
        "target_motif_strength": 0.25,
        "appropriate_motif_count": 0.15,
        "sequence_complexity": 0.15,
        "diversity": 0.15,
        "composition": 0.10,
        "training_similarity": 0.20,
    })

    def __post_init__(self) -> None:
        if self.min_length < 1 or self.max_length < self.min_length:
            raise ValueError("Invalid length range")
        if not 0 <= self.min_gc <= self.max_gc <= 1:
            raise ValueError("GC thresholds must satisfy 0 <= min_gc <= max_gc <= 1")
        if self.max_homopolymer < 1 or self.repeat_k < 1 or self.max_repeat_count < 1:
            raise ValueError("Repeat and homopolymer settings must be positive")
        if self.complexity_k < 1:
            raise ValueError("complexity_k must be positive")
        if self.motif_scan_step < 1:
            raise ValueError("motif_scan_step must be positive")
        if not 0 <= self.max_repeat_fraction <= 1 or not 0 <= self.min_complexity <= 1:
            raise ValueError("Fraction and complexity thresholds must be between 0 and 1")
        if not 0 <= self.duplicate_similarity <= 1:
            raise ValueError("duplicate_similarity must be between 0 and 1")
        if any(not math.isfinite(value) for value in self.score_weights.values()):
            raise ValueError("Score weights must be finite")
        known_weights = {
            "target_motif_strength",
            "appropriate_motif_count",
            "sequence_complexity",
            "diversity",
            "composition",
            "training_similarity",
        }
        unknown_weights = sorted(set(self.score_weights) - known_weights)
        if unknown_weights:
            raise ValueError(f"Unknown score weight component(s): {', '.join(unknown_weights)}")
        if any(value < 0 for value in self.score_weights.values()) or sum(self.score_weights.values()) <= 0:
            raise ValueError("Score weights must be non-negative and have positive total")


@dataclass
class CandidateResult:
    id: str
    sequence: str
    passed: bool
    issues: list[Issue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    motif_hits: list[MotifHit] = field(default_factory=list)
    duplicate_matches: list[DuplicateMatch] = field(default_factory=list)
    score: float = 0.0
    score_components: dict[str, float] = field(default_factory=dict)
    score_explanation: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
