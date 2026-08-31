"""Command-line interface."""

from __future__ import annotations

import argparse
import json

from .io import load_sequences, result_to_report, write_report
from .models import QCConfig
from .motifs import load_motifs
from .pipeline import run_qc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="QC and score generated regulatory DNA sequences")
    parser.add_argument("--input", required=True, help="Candidate FASTA, CSV, or JSON file")
    parser.add_argument("--training", help="Optional training FASTA, CSV, or JSON file")
    parser.add_argument("--motifs", help="Optional JSON PWM motif file")
    parser.add_argument("--output", required=True, help="Output JSON report path")
    parser.add_argument("--min-length", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=500)
    parser.add_argument("--min-gc", type=float, default=0.30)
    parser.add_argument("--max-gc", type=float, default=0.70)
    parser.add_argument("--max-homopolymer", type=int, default=8)
    parser.add_argument("--repeat-k", type=int, default=8)
    parser.add_argument("--max-repeat-count", type=int, default=3)
    parser.add_argument("--max-repeat-fraction", type=float, default=0.35)
    parser.add_argument("--min-complexity", type=float, default=0.55)
    parser.add_argument("--duplicate-similarity", type=float, default=0.95)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = QCConfig(
            min_length=args.min_length,
            max_length=args.max_length,
            min_gc=args.min_gc,
            max_gc=args.max_gc,
            max_homopolymer=args.max_homopolymer,
            repeat_k=args.repeat_k,
            max_repeat_count=args.max_repeat_count,
            max_repeat_fraction=args.max_repeat_fraction,
            min_complexity=args.min_complexity,
            duplicate_similarity=args.duplicate_similarity,
        )
        candidates = load_sequences(args.input)
        if not candidates:
            raise ValueError("Candidate input contains no sequences")
        training = load_sequences(args.training) if args.training else []
        motifs = load_motifs(args.motifs) if args.motifs else []
        results, _ = run_qc(candidates, motifs=motifs, training_records=training, config=config)
        report = result_to_report(results, config, len(motifs))
        write_report(args.output, report)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
