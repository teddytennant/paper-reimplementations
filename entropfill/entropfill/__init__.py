"""EntroPrefill, arXiv 2610.09757.

Sink-isolated Rényi pooling (equations 3-5), null-calibrated chunk
scores (8)-(9), the mixture-to-row deletion envelope (Theorem 4.1),
and the feasible scan of Section 4.2. No model, no RAG stack.
"""

from entropfill.envelope import deletion_envelope, envelope_holds
from entropfill.pooling import collision, pool, renyi2, sink_mass
from entropfill.scan import chunk_scores, feasible_scan
from entropfill.weights import head_weights

__all__ = [
    "chunk_scores",
    "collision",
    "deletion_envelope",
    "envelope_holds",
    "feasible_scan",
    "head_weights",
    "pool",
    "renyi2",
    "sink_mass",
]
