import os
import pickle
import torch


def save_checkpoint(model, cfg: dict, epoch: int, val_loss: float, label_to_idx: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if isinstance(model, torch.nn.Module):
        torch.save(
            {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": cfg,
                "label_to_idx": label_to_idx,
            },
            path,
        )
    else:
        # non-torch models (e.g. the Markov baseline) -> pickle
        with open(path, "wb") as f:
            pickle.dump(
                {"model": model, "epoch": epoch, "val_loss": val_loss, "config": cfg, "label_to_idx": label_to_idx},
                f,
            )


def load_checkpoint(path: str, model=None):
    if path.endswith(".pt"):
        # .pt files are always saved with torch.save, which is NOT plain
        # pickle (it wraps tensor storage with persistent IDs) — always use
        # torch.load for these regardless of whether a model instance was
        # passed in yet.
        ckpt = torch.load(path, map_location="cpu")
        if model is not None:
            model.load_state_dict(ckpt["model_state"])
            return model, ckpt
        # caller doesn't have a model instance yet (e.g. needs num_conditions
        # from ckpt["label_to_idx"] first) -- hand back the raw checkpoint
        # dict so they can build the model, then load_state_dict themselves.
        return None, ckpt
    with open(path, "rb") as f:
        ckpt = pickle.load(f)
    return ckpt["model"], ckpt
