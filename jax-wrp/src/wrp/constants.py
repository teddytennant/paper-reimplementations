"""Shared constants from arXiv:2609.09883."""

EPS = 1e-12

# Attention output and MLP down-projection, used for linear CKA (Eq. 1–2).
PROJ_KEYS = ("o", "down")

# Seven projections whose Frobenius norms form the scale descriptor (Eq. 3).
SCALE_KEYS = ("q", "k", "v", "o", "gate", "up", "down")

KMEANS_SEED = 0
