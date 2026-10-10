import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402

from scr.cluster import anchor_cluster, fingerprints
from scr.select import augment_indices, select_indices
from scr.attend import (
    per_subchunk_select,
    scr_attention,
    selection_calls,
    sparse_attention,
)

__all__ = [
    "anchor_cluster",
    "augment_indices",
    "fingerprints",
    "per_subchunk_select",
    "scr_attention",
    "select_indices",
    "selection_calls",
    "sparse_attention",
]

del jax
