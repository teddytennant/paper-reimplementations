# jax-assert-sampler

ASSERT in JAX: a training-free DDIM sampler that spends stochasticity early and anneals
it to zero, for diffusion models whose weights are perturbed by fixed analog
compute-in-memory error (arXiv:2609.00955).

## install

    uv venv
    uv pip install -e ".[test]"

## run

    python -m assert_sampler
    pytest

The noise model is the paper's parametric one, sigma_W^2 = alpha_W |W| + b_W with a
per-deployment spatial map, not error histograms measured from eFlash chips. Evaluation
is a 2D eight-mode mixture scored by energy distance, not CIFAR-10 or 256x256 FID, and
the network is a three-layer MLP. Equation 3's xhat_0 is clipped, which the paper does
not mention but the public checkpoints do. One result does not carry over: resampling
the perturbation every step shrinks ASSERT's absolute gain over DDIM by 3x but leaves
the ratio alone, so some of the gain here is the self-correction any stochastic sampler
gives against a wrong score. The decorrelation the paper argues for is measured
directly, on the cross terms of Equation 10, in tests/test_analysis.py.
