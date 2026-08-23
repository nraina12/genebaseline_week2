#helps isolate what difference conditional model can provide

import argparse
import json
import os
import random
import time
import yaml
import torch

from seed import set_seed
from checkpoint import load_checkpoint
from model import ConditionalMarkovModel, ConditionalLSTM
from encoding import BASES, indices_to_sequence, BASE_TO_IDX


def gc_content(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / max(len(seq), 1)


def generate_random(n: int, length: int) -> list:
    records = []
    for _ in range(n):
        seq = "".join(random.choice(BASES) for _ in range(length))
        records.append({
            "sequence": seq,
            "condition": None,
            "model_probability": None,
            "probability_type": "n/a_random",
            "metadata": {"length": len(seq), "gc_content": round(gc_content(seq), 4)},
            "generation_settings": {"model_type": "random_baseline"},
        })
    return records


def dinucleotide_shuffle(seq: str) -> str:
    """Shuffle while roughly preserving local (dinucleotide) composition —
    a stronger control than a naive single-base shuffle. Falls back to a
    simple base shuffle if the sequence is too short."""
    if len(seq) < 4:
        chars = list(seq)
        random.shuffle(chars)
        return "".join(chars)
    dinucs = [seq[i:i + 2] for i in range(0, len(seq) - 1, 2)]
    random.shuffle(dinucs)
    shuffled = "".join(dinucs)
    if len(seq) % 2 == 1:
        shuffled += seq[-1]
    return shuffled


def generate_shuffled(source_sequences: list) -> list:
    """source_sequences: list of real sequences (e.g. from Tisha's dataset
    once available, or from your own generated set) to shuffle."""
    records = []
    for seq in source_sequences:
        shuf = dinucleotide_shuffle(seq)
        records.append({
            "sequence": shuf,
            "condition": None,
            "model_probability": None,
            "probability_type": "n/a_shuffled",
            "metadata": {
                "length": len(shuf),
                "gc_content": round(gc_content(shuf), 4),
                "source_sequence_gc_content": round(gc_content(seq), 4),
            },
            "generation_settings": {"model_type": "shuffled_baseline", "method": "dinucleotide_shuffle"},
        })
    return records


def generate_unconditional_markov(cfg, train_sequences: list, n: int, length: int) -> list:
    """Fit a single Markov chain over ALL training sequences pooled together,
    ignoring condition labels entirely -- this is the "no conditioning
    signal" counterpart to the per-condition ConditionalMarkovModel."""
    pooled_model = ConditionalMarkovModel(order=cfg["model"]["order"])
    # cond=0 for everything -- pools all sequences into a single bucket
    pooled_model.fit(train_sequences, [0] * len(train_sequences))

    records = []
    for _ in range(n):
        seq = pooled_model.generate(0, length, temperature=cfg["generate"]["temperature"])
        log_prob = pooled_model.sequence_log_prob(seq, 0)
        records.append({
            "sequence": seq,
            "condition": None,
            "model_probability": log_prob,
            "probability_type": "log_likelihood_unconditional",
            "metadata": {"length": len(seq), "gc_content": round(gc_content(seq), 4)},
            "generation_settings": {
                "model_type": "markov_unconditional",
                "order": cfg["model"]["order"],
                "seed": cfg["seed"],
            },
        })
    return records


@torch.no_grad()
def generate_unconditional_lstm(model, cfg, device, n: int, length: int) -> list:
    """Reuses the trained conditional LSTM but samples a random condition
    index for every step (rather than holding one fixed), which washes out
    any consistent conditioning signal -- a quick proxy for "unconditional"
    without training a second model from scratch. If you want a stricter
    control, train a second LSTM with cond_emb removed/zeroed."""
    num_conditions = model.cond_emb.num_embeddings
    records = []
    for _ in range(n):
        fake_cond = random.randrange(num_conditions)
        idx_seq = model.generate(fake_cond, length, device, temperature=cfg["generate"]["temperature"])
        seq = indices_to_sequence(idx_seq)
        records.append({
            "sequence": seq,
            "condition": None,
            "model_probability": None,
            "probability_type": "n/a_unconditional_lstm",
            "metadata": {"length": len(seq), "gc_content": round(gc_content(seq), 4)},
            "generation_settings": {
                "model_type": "lstm_unconditional_proxy",
                "note": "condition randomized per sample to wash out conditioning signal",
                "temperature": cfg["generate"]["temperature"],
                "seed": cfg["seed"],
            },
        })
    return records


def main(config_path: str, checkpoint_path: str, which: list):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = cfg["generate"]["num_samples_per_condition"]
    length = cfg["data"]["pad_to"]
    out_dir = cfg["generate"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    all_records = {}

    if "random" in which:
        all_records["random"] = generate_random(n=n * 4, length=length)  # match total conditional volume

    if "shuffled" in which:
        # shuffle the sequences from the main conditional generation run if
        # available, otherwise fall back to freshly generated random ones
        candidates = []
        for fname in sorted(os.listdir(out_dir)):
            if fname.startswith("generated_") and fname.endswith(".jsonl"):
                with open(os.path.join(out_dir, fname)) as f:
                    for line in f:
                        candidates.append(json.loads(line)["sequence"])
                break
        if not candidates:
            candidates = [r["sequence"] for r in generate_random(n=n * 4, length=length)]
        all_records["shuffled"] = generate_shuffled(candidates)

    if "unconditional" in which:
        if cfg["model"]["type"] == "markov":
            from dataset import load_and_split
            split = load_and_split(cfg)
            train_sequences = split.train[cfg["data"]["seq_col"]].tolist()
            all_records["unconditional"] = generate_unconditional_markov(cfg, train_sequences, n=n * 4, length=length)
        else:
            model, ckpt = load_checkpoint(checkpoint_path, model=None)
            if model is not None:
                model.to(device)
                all_records["unconditional"] = generate_unconditional_lstm(model, cfg, device, n=n * 4, length=length)

    for name, records in all_records.items():
        out_path = os.path.join(out_dir, f"{name}_{int(time.time())}.jsonl")
        with open(out_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(records)} {name} sequences to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--which", nargs="+", default=["random", "shuffled", "unconditional"],
                         choices=["random", "shuffled", "unconditional"])
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.which)
