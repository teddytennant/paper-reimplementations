BridgeMem (https://arxiv.org/abs/2609.29268) as a JAX kernel: causal dyadic events are encoded by relation, direction, and recency (eq. 1), turned into a likelihood-ratio residual from a neural reader and a count reader (eqs. 2–4), and added to frozen backbone log scores through the uncertainty gate (eq. 5).

Install: `pip install -e .`

Run: `pytest` and `python train.py`

What does not match the paper: dummy frozen scorer instead of the released full-vocabulary backbone, no ICEWS/GDELT/WIKI dumps, no chronological 90/10 epoch selection or λ grid, Laplace add-1 on count probabilities (smoothing constant not displayed), toy 8-entity graph only.
