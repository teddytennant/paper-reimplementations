JAX library for Direct Conditional Transition Sampling (DCTS) on linear inverse problems (https://arxiv.org/abs/2609.14596).

Implements Algorithm 1 exactly under the paper's variance-exploding (VE) kernel α_t ≡ 1: EDM outer noise levels, inner path R_s with source scale σ_src, denoiser-compatible sufficient statistic (Eq. 7), covariance c_s = κ v_s, linear covariance-scaled operator update (Eq. 10), Euler integration of the conditional transition velocity (Eqs. 12–13), and direct transport of source noise to the next noisy state. Also exposes the general VP identity α_t² + σ_t² = 1 from the paper's kernel (Eq. 2). Plug-in denoisers are closed-form MMSE maps for a Gaussian prior. Forward operators are dense Gaussian, inpainting masks, and box downsampling. Tests and smoke run on tiny synthetic vectors.

```
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
JAX_PLATFORMS=cpu python -m pytest -q
python scripts/smoke.py
```

What does not match the paper: no FFHQ/ImageNet, no pretrained diffusion, no 16.8× runtime table, tiny synthetic inverse problems only. Algorithm 1 is VE (α_t ≡ 1); the VP schedule is the Eq. 2 identity, not a second sampler. No phase retrieval, motion deblur, or colorization. Eq. 11 (nonlinear Gauss–Newton) is omitted. Measurement updates use a dense linear solve rather than conjugate gradient. Default test schedules use fewer than L = 150 outer steps and smaller σ_max than 100.
