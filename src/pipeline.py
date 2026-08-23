""" 
# integrated Model V1 pipeline: one command that runs dataset loading 
# through condition encoding, training, validation, checkpointing, 
# evaluation, Sanjeet's QC (via qc_interface.py), and writes a standardized
# result file for downstream use

# wraps the existing, individually-tested scripts (train.py, generate.py,
# evaluate.py)

Entrypoint:
    python src/pipeline.py --config configs/baseline.yaml

# NOTE: qc_interface.py is a placeholder for Sanjeet's QC module. Once his
# can run against toy data now to verify the wiring itself
"""

import argparse
import json
import os
import time
import yaml
import torch
from typing import cast

from seed import set_seed
from dataset import load_and_split, SequenceDataset
from model import build_model
from model import ConditionalMarkovModel
from checkpoint import save_checkpoint, load_checkpoint
from qc_interface import run_qc
import train as train_module
import generate as generate_module


def run_pipeline(config_path: str) -> str:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    run_id = int(time.time())
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. dataset + condition encoding 
    split = load_and_split(cfg)
    label_to_idx = split.label_to_idx
    num_conditions = len(label_to_idx)
    print(f"[pipeline] {len(split.train)}/{len(split.val)}/{len(split.test)} "
          f"train/val/test, {num_conditions} conditions: {list(label_to_idx.keys())}")

    # training + validation + checkpointing 
    model = build_model(num_conditions, cfg)
    if cfg["model"]["type"] == "markov":
        train_seqs = split.train[cfg["data"]["seq_col"]].tolist()
        train_conds = split.train[cfg["data"]["label_col"]].map(label_to_idx).tolist()
        markov_model = cast(ConditionalMarkovModel, model)
        markov_model.fit(train_seqs, train_conds)
        ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], f"model_v1_{run_id}.pkl")
        save_checkpoint(model, cfg, epoch=0, val_loss=float("nan"), label_to_idx=label_to_idx, path=ckpt_path)
    else:
        from torch.utils.data import DataLoader
        from seed import seed_worker
        train_ds = SequenceDataset(split.train, label_to_idx, cfg)
        val_ds = SequenceDataset(split.val, label_to_idx, cfg)
        g = torch.Generator().manual_seed(cfg["seed"])
        train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                                   worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False)
        if isinstance(model, torch.nn.Module):
            model.to(device)
        train_module.train_lstm(model, cfg, train_loader, val_loader, device, label_to_idx)
        ckpt_path = os.path.join(cfg["train"]["checkpoint_dir"], "lstm_best.pt")

    print(f"[pipeline] checkpoint saved: {ckpt_path}")

    # 3. generate candidate sequences 
    if cfg["model"]["type"] == "markov":
        loaded_model, ckpt = load_checkpoint(ckpt_path)
        loaded_model = cast(ConditionalMarkovModel, loaded_model)
        records = generate_module.generate_markov(loaded_model, cfg, ckpt["label_to_idx"])
    else:
        _, ckpt = load_checkpoint(ckpt_path, model=None)
        from model import ConditionalLSTM
        loaded_model = ConditionalLSTM(len(ckpt["label_to_idx"]), ckpt["config"])
        loaded_model.load_state_dict(ckpt["model_state"])
        loaded_model.to(device)
        records = generate_module.generate_lstm(loaded_model, cfg, ckpt["label_to_idx"], device)

    print(f"[pipeline] generated {len(records)} candidate sequences")

    # 4. QC pass 
    for i, rec in enumerate(records):
        rec["id"] = i
        qc_result = run_qc(rec["sequence"], rec["condition"], rec,
                       min_len=cfg["data"]["min_len"], max_len=cfg["data"]["max_len"])
        rec["qc_passed"] = qc_result.passed
        rec["qc_flags"] = qc_result.flags
        rec["qc_scores"] = qc_result.scores

    n_passed = sum(1 for r in records if r["qc_passed"])
    print(f"[pipeline] QC: {n_passed}/{len(records)} passed (NOTE: qc_interface.py is a "
          f"placeholder pending Sanjeet's actual QC module ")

    # 5. standardized result file 
    out_dir = cfg["generate"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"model_v1_results_{run_id}.jsonl")
    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    manifest = {
        "run_id": run_id,
        "config_path": config_path,
        "config": cfg,
        "checkpoint_path": ckpt_path,
        "n_generated": len(records),
        "n_qc_passed": n_passed,
        "results_path": out_path,
    }
    manifest_path = os.path.join(out_dir, f"model_v1_manifest_{run_id}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[pipeline] results: {out_path}")
    print(f"[pipeline] manifest (for reproducibility): {manifest_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    run_pipeline(args.config)
