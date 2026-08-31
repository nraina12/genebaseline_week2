"""FASTA, CSV, JSON, and JSONL input plus JSON report output."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .models import CandidateResult, SequenceRecord


def load_sequences(path: str | Path) -> list[SequenceRecord]:
    path = Path(path)
    if path.suffix.lower() in {".fa", ".fasta", ".fas"}:
        return _load_fasta(path)
    if path.suffix.lower() == ".csv":
        return _load_csv(path)
    if path.suffix.lower() == ".json":
        return _load_json(path)
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl(path)
    raise ValueError(f"Unsupported sequence input type: {path.suffix}")


def write_report(path: str | Path, report: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def result_to_report(results: list[CandidateResult], config: Any, motif_count: int) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "config": config.__dict__,
        "motif_count": motif_count,
        "summary": {
            "candidate_count": len(results),
            "passed_count": sum(result.passed for result in results),
            "failed_count": sum(not result.passed for result in results),
            "mean_score": round(sum(result.score for result in results) / len(results), 3) if results else 0.0,
        },
        "candidates": [result.to_dict() for result in results],
    }


def _load_fasta(path: Path) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    current_id: str | None = None
    current_metadata: dict[str, str] = {}
    chunks: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_id is not None:
                records.append(SequenceRecord(current_id, "".join(chunks), current_metadata))
            header = line[1:].strip()
            if not header:
                raise ValueError(f"FASTA header on line {line_number} has no ID")
            header_token = header.split()[0]
            parts = header_token.split("|")
            current_id = parts[0]
            if not current_id:
                raise ValueError(f"FASTA header on line {line_number} has no ID")
            current_metadata = {}
            for part in parts[1:]:
                if "=" in part:
                    key, value = part.split("=", 1)
                    if key:
                        current_metadata[key] = value
            chunks = []
        else:
            if current_id is None:
                raise ValueError(f"FASTA sequence appears before a header on line {line_number}")
            chunks.append(line)
    if current_id is not None:
        records.append(SequenceRecord(current_id, "".join(chunks), current_metadata))
    return records


def _load_csv(path: Path) -> list[SequenceRecord]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "id" not in reader.fieldnames or "sequence" not in reader.fieldnames:
            raise ValueError("CSV input requires id and sequence columns")
        records: list[SequenceRecord] = []
        for row_number, row in enumerate(reader, start=2):
            if row.get(None):
                raise ValueError(f"CSV row {row_number} contains an extra value")
            record_id = row.get("id")
            sequence = row.get("sequence")
            if record_id is None or not record_id.strip():
                raise ValueError(f"CSV row {row_number} is missing an ID")
            if sequence is None:
                raise ValueError(f"CSV row {row_number} is missing a sequence")
            metadata = {
                key: value
                for key, value in row.items()
                if key is not None and key not in {"id", "sequence"}
            }
            records.append(SequenceRecord(record_id, sequence, metadata))
        return records


def _load_json(path: Path) -> list[SequenceRecord]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        payload = [
            dict(value, id=key) if isinstance(value, dict) else {"id": key, "sequence": value}
            for key, value in payload.items()
        ]
    if not isinstance(payload, list):
        raise ValueError("JSON sequence input must be a list or an ID-to-sequence mapping")
    records: list[SequenceRecord] = []
    for item in payload:
        if not isinstance(item, dict) or "id" not in item or "sequence" not in item:
            raise ValueError("Each JSON sequence record requires id and sequence")
        if item["sequence"] is None:
            raise ValueError("JSON sequence record is missing a sequence")
        records.append(SequenceRecord(
            str(item["id"]),
            str(item["sequence"]),
            {key: value for key, value in item.items() if key not in {"id", "sequence"}},
        ))
    return records


def _load_jsonl(path: Path) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    record_number = 0
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        record_number += 1
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"JSONL line {line_number} contains invalid JSON: {error.msg}"
            ) from error
        if not isinstance(item, dict):
            raise ValueError(f"JSONL line {line_number} must contain an object")
        if "sequence" not in item or item["sequence"] is None:
            raise ValueError(f"JSONL line {line_number} is missing a sequence")
        raw_id = item.get("id")
        record_id = str(raw_id).strip() if raw_id is not None else ""
        if not record_id:
            record_id = f"{path.stem}_{record_number:06d}"
        records.append(SequenceRecord(
            record_id,
            str(item["sequence"]),
            {key: value for key, value in item.items() if key not in {"id", "sequence"}},
        ))
    return records
