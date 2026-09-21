"""MDL hyperparameters from arXiv:2609.22043, Table 2 and Fig. 4."""

from __future__ import annotations

D = 16
SUBSPACE_DIMS = (5, 5, 6)  # (d_wm, d_r, d_a)
SIGMA = 0.20
EPS_C = 0.10
GAINS = (1.0, 0.7, 0.5)  # (s_wm, s_r, s_a)
THETA = (0.6, 0.4, 0.2)  # Fig. 4, QR seed 42
QR_SEED = 42
BIAS = 0.0
RELEVANCE_COS_THRESHOLD = 0.3

# Paper fits nmin/nmax as min/max ||vmeta|| on an 80-question calibration split.
# Defaults here are a compact synthetic band around fused norms of high-(M, R)
# inputs under seed-42 QR: low-risk high-relevance sits near Active, while
# (M, R, A) = (0.9, 0.9, 0.85) falls below θ1 (Silent / Opt-Out), matching
# Fig. 4's contraction of the Active region as risk rises. Override via args.
NMIN = 2.62
NMAX = 3.32

ALPHA_FLOOR = 0.3
ALPHA_WEIGHT = 0.7
