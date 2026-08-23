"""Two ways to turn a DNA sequence into a feature vector, behind one
interface, so the comparison in compare_representations.py can swap
between them without touching the rest of the pipeline:

  - "simple": the representation your model already uses -- one-hot /
    learned nn.Embedding over bases (see encoding.py + ConditionalLSTM's
    base_emb). No pretraining, fast, no external dependencies.
  - "enformer": Enformer (Avsec et al. 2021) pretrained genomic
    representation. MUCH heavier: expects a 196,608bp input window and has
    ~250M parameters. Practical notes below.

IMPORTANT / NOT YET RUNNABLE HERE: I could not actually execute the
Enformer path in this environment -- it needs the `enformer-pytorch`
package plus downloading multi-GB pretrained weights, and realistically a
GPU to run in reasonable time. Treat get_enformer_embedding() as a
correctly-structured but UNTESTED stub; validate it on lab hardware before
trusting its output.

Known integration wrinkle worth flagging to the team: Enformer's receptive
field (196,608bp) is vastly larger than your 200-500bp regulatory
sequences. Two common ways to handle this mismatch:
  (a) Pad each sequence with real flanking genomic sequence (if Tisha's
      coordinates let you pull the actual up/downstream context) -- more
      biologically valid, more data-engineering work.
  (b) Pad with N / random sequence on either side -- fast, no extra data
      needed, but the embedding at the region of interest may be affected
      by what you padded it with. Enformer's own literature generally
      assumes real genomic context, so (a) is preferable if feasible.
This module defaults to (b) for a quick comparison and logs a warning,
but flip pad_with_real_context=True (and supply flanks) once you have
real coordinates from Tisha to do it properly.
"""

import importlib
import warnings
import torch
import torch.nn as nn

from encoding import sequence_to_one_hot

ENFORMER_CONTEXT_LENGTH = 196_608


class SimpleEmbeddingRepresentation(nn.Module):
    """Wraps the same base embedding scheme already used in model.py's
    ConditionalLSTM, exposed standalone so it can be probed/compared
    the same way as the Enformer path."""

    def __init__(self, embedding_dim: int = 32):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.base_emb = nn.Embedding(5, embedding_dim)  # A,C,G,T,PAD

    def forward(self, seq_indices: torch.Tensor) -> torch.Tensor:
        """seq_indices: (B, L) -> (B, L, embedding_dim). Callers typically
        mean-pool over L to get a fixed-size per-sequence vector for a
        downstream probe."""
        return self.base_emb(seq_indices)


def get_enformer_embedding(seq: str, pad_with_real_context: bool = False,
                            left_flank: str = "", right_flank: str = "") -> torch.Tensor:
    """UNTESTED STUB -- structured correctly per enformer-pytorch's expected
    usage, but not run in this environment. Verify on lab hardware.

    Returns a pooled embedding vector for the input sequence's central
    region from Enformer's target-length output track dimension.
    """
    try:
        from enformer_pytorch import Enformer, seq_indices_to_one_hot, str_to_one_hot
    except ImportError as e:
        raise ImportError(
            "enformer-pytorch is not installed. On lab hardware: "
            "pip install enformer-pytorch (requires torch + significant "
            "disk/GPU for the pretrained weights)."
        ) from e

    seq = seq.upper()
    if pad_with_real_context:
        padded = (left_flank + seq + right_flank).upper()
    else:
        warnings.warn(
            "Padding Enformer input with N's, not real genomic flanking "
            "context -- results may not reflect Enformer's intended usage. "
            "See module docstring for details.",
            stacklevel=2,
        )
        pad_total = ENFORMER_CONTEXT_LENGTH - len(seq)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        padded = ("N" * pad_left) + seq + ("N" * pad_right)

    padded = padded[:ENFORMER_CONTEXT_LENGTH].ljust(ENFORMER_CONTEXT_LENGTH, "N")

    model = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
    model.eval()

    one_hot = str_to_one_hot(padded).unsqueeze(0)  # (1, L, 4)
    with torch.no_grad():
        # embeddings=True returns the pre-head representation rather than
        # the full track predictions -- what we actually want for a
        # "representation quality" comparison.
        _, embeddings = model(one_hot, return_embeddings=True)

    # embeddings shape is roughly (1, target_length, feature_dim);
    # mean-pool over the target/bin dimension for a fixed-size vector.
    return embeddings.mean(dim=1).squeeze(0)
