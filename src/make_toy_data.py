"""Run once to smoke-test the pipeline before real data from Tisha arrives:
    python src/make_toy_data.py
Writes data/sequences.csv with synthetic condition-biased sequences."""

import os
import random
import pandas as pd

random.seed(0)
BASES = "ACGT"
CONDITIONS = ["KRAS_MAPK_ERK", "HNF4G_FOXA1", "GATA6", "PTF1A_NEGATIVE"]


def biased_seq(length: int, bias_base: str, bias_strength: float) -> str:
    out = []
    for _ in range(length):
        if random.random() < bias_strength:
            out.append(bias_base)
        else:
            out.append(random.choice(BASES))
    return "".join(out)


rows = []
# distinct synthetic base-composition biases per real condition, purely so
# the toy data has SOME learnable structure per condition -- these bias
# choices are arbitrary placeholders and carry no biological meaning; swap
# data/sequences.csv for real labeled data as soon as it's available.
bias_map = {
    "KRAS_MAPK_ERK": "G",
    "HNF4G_FOXA1": "A",
    "GATA6": "T",
    "PTF1A_NEGATIVE": None,  # no bias, so it doesn't resemble the others
}
for cond, bias in bias_map.items():
    for _ in range(300):
        length = random.randint(200, 500)
        seq = biased_seq(length, bias, 0.35) if bias else "".join(random.choice(BASES) for _ in range(length))
        rows.append({"sequence": seq, "condition": cond})

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path = os.path.join(repo_root, "data", "sequences.csv")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"Wrote {out_path} with", len(rows), "rows")