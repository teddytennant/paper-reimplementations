# jax-ugtl

JAX implementation of Uncertainty-aware Gradient-guided Trajectory Learning
and the algorithm-dependent learnability bounds from "Rethinking Learnability
in Offline Data-driven Optimization" (arXiv:2609.01493).

## install

    uv pip install -e .

## run

    python -m ugtl

Design-Bench is not here, so there is no ranking against the paper's 25
baselines. The pipeline runs on the negated Branin function of Figure 1 at a
CPU-sized scale, with an MLP denoiser instead of the temporal U-Net and an x0
clamp in the sampler that the paper does not use. The theory tests are empirical
checks that the measured ratio clears the alpha of Theorems 4, 5 and 6, not
proofs of them. The constructor reproduces the Figure 2 geometry against a
random-construction ablation, paths hundreds of times smoother by the
second-difference measure, but its candidates do not beat that ablation's; on a
2D task the proxy screen in Algorithm 6 is accurate enough to hide the
difference.
