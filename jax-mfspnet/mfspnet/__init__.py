"""MFSPNet: model-free surrogate PSO search over variable-length dense blocks.

A JAX implementation of arXiv:2609.02460. Section, equation and algorithm
numbers in the docstrings refer to that paper.
"""

from . import data, encoding, estimators, model, pso, ranking, search, signal_noise, train

__all__ = [
    "data",
    "encoding",
    "estimators",
    "model",
    "pso",
    "ranking",
    "search",
    "signal_noise",
    "train",
]
