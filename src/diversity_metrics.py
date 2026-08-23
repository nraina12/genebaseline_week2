#katelyn 4.4

import argparse
import json
import itertools
from typing import List


def load_sequences(path: str) -> List[str]:
    seqs = []
    with open(path) as f:
        for line in f:
            seqs.append(json.loads(line)["sequence"])
    return seqs


def percent_unique(seqs: List[str]) -> float:
    return 100.0 * len(set(seqs)) / max(len(seqs), 1)


def hamming_similarity(a: str, b: str) -> float:
    length = min(len(a), len(b))
    if length == 0:
        return 0.0
    matches = sum(1 for i in range(length) if a[i] == b[i])
    return matches / length


def pairwise_similarity_stats(seqs: List[str], max_pairs: int = 2000) -> dict:
    #caps num of pairs to compare
    pairs = list(itertools.combinations(range(len(seqs)), 2))
    if len(pairs) > max_pairs:
        import random
        random.seed(0)
        pairs = random.sample(pairs, max_pairs)

    sims = [hamming_similarity(seqs[i], seqs[j]) for i, j in pairs]
    if not sims:
        return {"mean": None, "max": None, "n_pairs_sampled": 0}
    return {
        "mean": sum(sims) / len(sims),
        "max": max(sims),
        "n_pairs_sampled": len(sims),
    }


def count_near_duplicates(seqs: List[str], similarity_threshold: float = 0.95, max_pairs: int = 2000) -> int:
    pairs = list(itertools.combinations(range(len(seqs)), 2))
    if len(pairs) > max_pairs:
        import random
        random.seed(0)
        pairs = random.sample(pairs, max_pairs)
    return sum(1 for i, j in pairs if hamming_similarity(seqs[i], seqs[j]) >= similarity_threshold)


def summarize(path: str) -> dict:
    seqs = load_sequences(path)
    return {
        "n_sequences": len(seqs),
        "percent_unique": round(percent_unique(seqs), 2),
        "pairwise_similarity": pairwise_similarity_stats(seqs),
        "near_duplicates_at_0.95_similarity": count_near_duplicates(seqs),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a .jsonl file with a 'sequence' field per line")
    args = parser.parse_args()
    result = summarize(args.input)
    print(json.dumps(result, indent=2))
