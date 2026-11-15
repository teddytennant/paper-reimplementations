"""Displayed constants from arXiv:2609.29268."""

K = 8
WINDOW = 65
# Logarithmic recency bin edges {0, 1, 4, 16, 64}. Five bins:
# (0, 1], (1, 4], (4, 16], (16, 64], (64, inf).
BIN_EDGES = (0, 1, 4, 16, 64)
N_BINS = 5
TAU = 100.0
EPS = 1e-12
EMBED_DIM = 64
HIDDEN = 128
LN_EPS = 1e-5
