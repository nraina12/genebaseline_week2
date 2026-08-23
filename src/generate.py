import argparse
import json
import os
import time
import yaml
import torch

from seed import set_seed
from checkpoint import load_checkpoint
from model import ConditionalMarkovModel, ConditionalLSTM
from encoding import indices_to_sequence


def generate_markov(model: ConditionalMarkovModel, cfg, label_to_idx):
    records = []
    length = cfg["data"]["pad_to"]
    n = cfg["generate"]["num_samples_per_condition"]
    temp = cfg["generate"]["temperature"]

    for label, cond_idx in label_to_idx.items():
        for _ in range(n):
            seq = model.generate(cond_idx, length, temperature=temp)
            log_prob = model.sequence_log_prob(seq, cond_idx)
            records.append({
                "sequence": seq,
                "condition": label,
                "model_probability": log_prob,   # log-likelihood under the fitted Markov model
                "probability_type": "log_likelihood",
                "metadata": {
                    "length": len(seq),
                    "gc_content": round((seq.count("G") + seq.count("C")) / max(len(seq), 1), 4),
                },
                "generation_settings": {
                    "model_type": "markov",
                    "order": cfg["model"]["order"],
                    "temperature": temp,
                    "seed": cfg["seed"],
                },
            })
    return records


@torch.no_grad()
def generate_lstm(model: ConditionalLSTM, cfg, label_to_idx, device):
    records = []
    length = cfg["data"]["pad_to"]
    n = cfg["generate"]["num_samples_per_condition"]
    temp = cfg["generate"]["temperature"]

    for label, cond_idx in label_to_idx.items():
        for _ in range(n):
            idx_seq = model.generate(cond_idx, length, device, temperature=temp)
            seq = indices_to_sequence(idx_seq)

            # approximate model probability: mean per-step confidence, teacher-forced
            logits = model(idx_seq.unsqueeze(0)[:, :-1], torch.tensor([cond_idx], device=device))
            probs = torch.softmax(logits, dim=-1)
            targets = idx_seq[1:].clamp(max=3)
            step_probs = probs[0, torch.arange(len(targets)), targets]
            mean_prob = step_probs.mean().item()

            records.append({
                "sequence": seq,
                "condition": label,
                "model_probability": mean_prob,
                "probability_type": "mean_next_base_prob",
                "metadata": {
                    "length": len(seq),
                    "gc_content": round((seq.count("G") + seq.count("C")) / max(len(seq), 1), 4),
                },
                "generation_settings": {
                    "model_type": "lstm",
                    "temperature": temp,
                    "seed": cfg["seed"],
                },
            })
    return records


def main(config_path: str, checkpoint_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg["model"]["type"] == "markov":
        model, ckpt = load_checkpoint(checkpoint_path)
        assert isinstance(model, ConditionalMarkovModel)
        records = generate_markov(model, cfg, ckpt["label_to_idx"])
    else:
        # First load just gets the raw checkpoint (incl. label_to_idx), since
        # we need the number of conditions before we can construct the model.
        _, ckpt = load_checkpoint(checkpoint_path, model=None)
        model = ConditionalLSTM(len(ckpt["label_to_idx"]), ckpt["config"])
        model.load_state_dict(ckpt["model_state"])
        model.to(device)
        records = generate_lstm(model, cfg, ckpt["label_to_idx"], device)

    out_dir = cfg["generate"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"generated_{int(time.time())}.jsonl")
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(records)} generated sequences to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    main(args.config, args.checkpoint)

