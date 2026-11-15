"""RouteRelay: event-triggered cross-layer reuse of sparse attention routes."""

from routerelay.relay import (
    Ledger,
    RelayInfo,
    anchor_margin,
    build_ledger,
    candidate_set,
    chunk_scores,
    miss_probability,
    miss_probability_bound,
    near_miss_sentinels,
    relay_layer,
    rescore_candidates,
    reuse_mask,
    topk_routes,
)
from routerelay.synthetic import StackResult, run_stack, toy_chunk_stack

__all__ = [
    "Ledger",
    "RelayInfo",
    "StackResult",
    "anchor_margin",
    "build_ledger",
    "candidate_set",
    "chunk_scores",
    "miss_probability",
    "miss_probability_bound",
    "near_miss_sentinels",
    "relay_layer",
    "rescore_candidates",
    "reuse_mask",
    "run_stack",
    "topk_routes",
    "toy_chunk_stack",
]
