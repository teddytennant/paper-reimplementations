"""MaskCode: a Tanner-graph-masked inner feedback Transformer (arXiv:2609.00715)."""

from .codes import LinearBlockCode, array_ldpc, bch, hamming74, single_parity_check, tree_code
from .model import MaskCode, MaskCodeConfig, run_batch
from .train import TrainConfig, train

__all__ = [
    "LinearBlockCode",
    "MaskCode",
    "MaskCodeConfig",
    "TrainConfig",
    "array_ldpc",
    "bch",
    "hamming74",
    "run_batch",
    "single_parity_check",
    "train",
    "tree_code",
]
