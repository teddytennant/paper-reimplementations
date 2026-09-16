JAX implementation of multi-agent cooperation-driven training (arXiv:2609.16917).

Install: `pip install -e ".[dev]"`

Run: `JAX_PLATFORMS=cpu python -m pytest -q` and `JAX_PLATFORMS=cpu python scripts/smoke.py`

What does not match the paper: tiny 2D Gaussian blobs, not MNIST/Fashion-MNIST/CIFAR-10; 1-hidden-layer MLPs (hidden 8 vs 32) rather than the paper's sizes. Paper eq. (5) is `w_k = softmax(-H_k / τ)`; `confidence()` is `1 - H / log(C)` in [0, 1] and weighted-average / simplicial consensus normalizes those scores (`softmax_weights` is eq. (5)). Eq. (11) writes a minus on local CE; training uses the positive CE of voter/majority and of eq. (14). Majority vote is one-hot argmax ballots with smallest-index tie-break; eq. (7) typesets a sum of raw softmax vectors. Combined loss is the multiplicative form of eq. (3), `L * (L + λ L^G)`, default `λ = 1` as in Table I. `τ` defaults to 1.0 (paper only states `τ > 0`). Smoke uses SGD, not Table IV's Adam. Simplicial consensus is all-to-all confidence-weighted averaging, not a graph 1-simplex / pairwise KL.
