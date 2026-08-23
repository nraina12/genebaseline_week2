"""Placeholder for Sanjeet's QC module. Everything downstream (pipeline.py)
calls run_qc() through this interface, so once his real spec/code arrives,
only this file needs to change -- nothing else in the pipeline does.

Current behavior is a stand-in: basic sanity checks only (valid bases,
correct length, non-empty). Replace the body of run_qc() with a call into
Sanjeet's actual module once he responds.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class QCResult:
    sequence_id: int
    passed: bool
    flags: List[str] = field(default_factory=list)
    scores: dict = field(default_factory=dict)


def run_qc(sequence: str, condition: str, metadata: dict, min_len: int = 200, max_len: int = 500) -> QCResult:
    """PLACEHOLDER -- replace with Sanjeet's actual QC call. min_len/max_len should be passed from
    the run's config (data.min_len / data.max_len) so this always matches
    whichever dataset is actually being used."""
    flags = []
    valid_bases = set("ACGT")
    if not sequence or not set(sequence.upper()).issubset(valid_bases):
        flags.append("invalid_bases")
    if not (min_len <= len(sequence) <= max_len):
        flags.append(f"length_out_of_range (got {len(sequence)}, expected [{min_len},{max_len}])")

    return QCResult(
        sequence_id=metadata.get("id", -1),
        passed=(len(flags) == 0),
        flags=flags,
        scores={},
    )