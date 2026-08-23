from collections import defaultdict, Counter
from typing import Dict, List
import random
import torch
import torch.nn as nn

from encoding import BASES, BASE_TO_IDX, IDX_TO_BASE, VOCAB_SIZE


def _new_context_counter():
    return defaultdict(Counter)


class ConditionalMarkovModel:

    def __init__(self, order: int = 3):
        self.order = order
        self.counts: Dict[int, Dict[str, Counter]] = defaultdict(_new_context_counter)
        self.start_contexts: Dict[int, List[str]] = defaultdict(list)

    def fit(self, sequences: List[str], conditions: List[int]) -> None:
        for seq, cond in zip(sequences, conditions):
            seq = seq.upper()
            if len(seq) <= self.order:
                continue
            self.start_contexts[cond].append(seq[: self.order])
            for i in range(len(seq) - self.order):
                context = seq[i:i + self.order]
                nxt = seq[i + self.order]
                if nxt in BASE_TO_IDX:
                    self.counts[cond][context][nxt] += 1

    def _sample_next(self, cond: int, context: str) -> str:
        counter = self.counts[cond].get(context)
        if not counter:
            return random.choice(BASES)
        bases, weights = zip(*counter.items())
        return random.choices(bases, weights=weights, k=1)[0]

    def generate(self, cond: int, length: int, temperature: float = 1.0) -> str:
        starts = self.start_contexts.get(cond) or ["".join(random.choices(BASES, k=self.order))]
        seq = random.choice(starts)
        while len(seq) < length:
            context = seq[-self.order:]
            seq += self._sample_next(cond, context)
        return seq[:length]

    def sequence_log_prob(self, seq: str, cond: int) -> float:
        import math
        seq = seq.upper()
        if len(seq) <= self.order:
            return float("nan")
        log_p = 0.0
        for i in range(len(seq) - self.order):
            context = seq[i:i + self.order]
            nxt = seq[i + self.order]
            counter = self.counts[cond].get(context)
            if not counter:
                p = 1.0 / 4
            else:
                total = sum(counter.values())
                p = counter.get(nxt, 0.5) / total  # +0.5 smoothing-ish floor
            log_p += math.log(max(p, 1e-8))
        return log_p


class ConditionalLSTM(nn.Module):
    #condition label is embedded and concatenated to every input
    def __init__(self, num_conditions: int, cfg: dict):
        super().__init__()
        m = cfg["model"]
        self.pad_to = cfg["data"]["pad_to"]
        self.base_emb = nn.Embedding(VOCAB_SIZE, m["embedding_dim"])
        self.cond_emb = nn.Embedding(num_conditions, m["embedding_dim"])
        self.lstm = nn.LSTM(
            input_size=m["embedding_dim"] * 2,
            hidden_size=m["hidden_dim"],
            num_layers=m["num_layers"],
            dropout=m["dropout"] if m["num_layers"] > 1 else 0.0,
            batch_first=True,
        )
        self.out = nn.Linear(m["hidden_dim"], 4)  # predict next base (A/C/G/T only)

    def forward(self, seq_idx: torch.Tensor, cond_idx: torch.Tensor) -> torch.Tensor:
        """seq_idx: (B, L) indices including PAD; cond_idx: (B,)
        Returns logits (B, L, 4) predicting the base at each position
        (teacher-forced, shifted by caller in the training loop)."""
        B, L = seq_idx.shape
        base_e = self.base_emb(seq_idx)                         # (B, L, E)
        cond_e = self.cond_emb(cond_idx).unsqueeze(1).expand(-1, L, -1)  # (B, L, E)
        x = torch.cat([base_e, cond_e], dim=-1)
        h, _ = self.lstm(x)
        return self.out(h)

    @torch.no_grad()
    def generate(self, cond_idx: int, length: int, device, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        cond_t = torch.tensor([cond_idx], device=device)
        seq = torch.full((1, 1), fill_value=BASE_TO_IDX["A"], dtype=torch.long, device=device)
        hidden = None
        for _ in range(length - 1):
            base_e = self.base_emb(seq[:, -1:])
            cond_e = self.cond_emb(cond_t).unsqueeze(1)
            x = torch.cat([base_e, cond_e], dim=-1)
            h, hidden = self.lstm(x, hidden)
            logits = self.out(h[:, -1, :]) / max(temperature, 1e-4)
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            seq = torch.cat([seq, nxt], dim=1)
        return seq.squeeze(0)


def build_model(num_conditions: int, cfg: dict):
    if cfg["model"]["type"] == "markov":
        return ConditionalMarkovModel(order=cfg["model"]["order"])
    elif cfg["model"]["type"] == "lstm":
        return ConditionalLSTM(num_conditions, cfg)
    raise ValueError(f"Unknown model type: {cfg['model']['type']}")
