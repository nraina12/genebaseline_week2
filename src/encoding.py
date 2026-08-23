from typing import List
import torch

BASES = ["A", "C", "G", "T"]
BASE_TO_IDX = {b: i for i, b in enumerate(BASES)}
IDX_TO_BASE = {i: b for b, i in BASE_TO_IDX.items()}
PAD_IDX = 4          # index used for padding
PAD_TOKEN = "N"
VOCAB_SIZE = 5        # A, C, G, T, PAD/N


def clean_sequence(seq: str) -> str:
    #func assumes seq is already ACGT/N
    return seq.strip().upper()


def pad_or_truncate(seq: str, length: int) -> str:
    seq = clean_sequence(seq)
    if len(seq) >= length:
        return seq[:length]
    return seq + PAD_TOKEN * (length - len(seq))


def sequence_to_indices(seq: str, length: int) -> torch.Tensor:
    #converts DNA sequence to atensor of indices
    seq = pad_or_truncate(seq, length)
    idx = [BASE_TO_IDX.get(b, PAD_IDX) for b in seq]
    return torch.tensor(idx, dtype=torch.long)


def sequence_to_one_hot(seq: str, length: int) -> torch.FloatTensor:
    #converts DNA sequence to one-hot encoding
    idx = sequence_to_indices(seq, length)
    one_hot = torch.zeros(length, 4, dtype=torch.float)
    valid = idx < 4
    one_hot[torch.arange(length)[valid], idx[valid]] = 1.0
    return one_hot


def indices_to_sequence(idx: torch.Tensor) -> str:
    return "".join(IDX_TO_BASE.get(int(i), PAD_TOKEN) for i in idx)


def reverse_complement(seq: str) -> str:
    comp = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
    return "".join(comp.get(b, "N") for b in reversed(seq.upper()))


def batch_one_hot(seqs: List[str], length: int) -> torch.FloatTensor:
    return torch.stack([sequence_to_one_hot(s, length) for s in seqs])
