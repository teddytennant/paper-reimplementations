"""Tree-structured training targets for rollouts under context compression.

A JAX implementation of arXiv:2609.00865, "MemoryWalker: Stop Training Agents on
Contexts They Never Saw": the trajectory tree an eviction schedule induces, the
two exact linearizations of it (LogitTree and the packed 4D attention mask), the
two flattenings that get it wrong, and SDCC.
"""

from .exact import logittree_loss, packed4d_loss
from .model import Config, default_config, init_params
from .naive import naive_compressed_loss, naive_full_loss
from .reference import materialized_tree_loss
from .sdcc import residual_kl, sdcc_loss, total_variation
from .tree import Eviction, Rollout, running_example

__all__ = [
    "Eviction", "Rollout", "running_example",
    "Config", "default_config", "init_params",
    "materialized_tree_loss", "logittree_loss", "packed4d_loss",
    "naive_compressed_loss", "naive_full_loss",
    "sdcc_loss", "residual_kl", "total_variation",
]
