from typing import List, Optional

from sanjeet_qc import run_qc as _sanjeet_run_qc
from sanjeet_qc import SequenceRecord, QCConfig
from sanjeet_qc.motifs import load_motifs


def run_qc_batch(records: List[dict], min_len: int, max_len: int,
                  motifs_path: str = "data/motifs_sanjeet.json",
                  training_sequences: Optional[List[str]] = None) -> tuple:
    """records: list of dicts with at least 'sequence' and 'condition' keys
    (the format generate.py / pipeline.py already produce). Returns
    (results, summary) from Sanjeet's run_qc -- CandidateResult objects,
    one per input record, in order, plus his summary dict.
    """
    motifs = load_motifs(motifs_path)
    config = QCConfig(min_length=min_len, max_length=max_len)

    candidates = [
        SequenceRecord(id=str(i), sequence=rec["sequence"],
                        metadata={"condition": rec.get("condition")})
        for i, rec in enumerate(records)
    ]
    training_records = [
        SequenceRecord(id=f"train_{i}", sequence=seq, metadata={})
        for i, seq in enumerate(training_sequences or [])
    ]

    results, summary = _sanjeet_run_qc(candidates, motifs=motifs,
                                        training_records=training_records, config=config)
    return results, summary
