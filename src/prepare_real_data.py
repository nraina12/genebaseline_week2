import argparse
import os
import pandas as pd


def convert(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_excel(input_path)
    df = df.dropna(subset=["Gene", "Sequence (FASTA)", "TF / Pathway Condition"])

    out = pd.DataFrame({
        "sequence": df["Sequence (FASTA)"].str.strip().str.upper(),
        "condition": df["TF / Pathway Condition"].str.strip(),
        "gene": df["Gene"],
        "strand": df.get("Strand"),
        "confidence": df.get("Confidence Level"),
        "motif_annotation": df.get("TF Motif Scores (JASPAR)"),
        "cell_context": df.get("Cell Context"),
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out.to_csv(output_path, index=False)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/real_sequences.csv")
    args = parser.parse_args()
    out = convert(args.input, args.output)
    print(f"Wrote {len(out)} real sequences to {args.output}")
    print(out["condition"].value_counts())
