QuAKE Kalman corrector for quantized diffusion sampling ([arXiv:2609.21407](https://arxiv.org/abs/2609.21407)): online linear-Gaussian estimator over a denoiser-output window, posterior mean fed to the sampler.

```
pip install -e . && pytest
```

```
python -m quake.demo
```

## What does not match the paper

- no SD3/FLUX, no W4A4, no real T2I
- no DPM-Solver++ or UniPC (`Ψ_i`); the library is the Kalman corrector only
- no SVDQuant / network quantization of a UNet or DiT
- no MJHQ-30K, sDCI, FID, KID, CLIP, CLIP-IQA, or ImageReward
- window length is called `H` here (`N` in the paper)
- predict is `m ← A m + b + G μ` with Lagrange affine bias `b` kept separate from the process-residual mean `μ` (paper eqs 9–11 fold `μ` into `b = μ e_1` and use zero-mean `η`)
- covariance update uses the simple form `P ← (I − K H) P` rather than `P − K S K^⊤` (eq 23 / Algorithm 1); they are algebraically equivalent
- default Lagrange-in-`λ` acts in the output domain; optional affine `x0 = α f + β` maps are provided, but there is no learned denoiser parameterization or noise-schedule conversion
- channel-wise independent 1D filters on the window (paper Appendix C granularity), not full spatial or cross-channel covariance
- calibration is JAX moments over synthetic paired tensors, not 1,024 real images
- no classifier-free guidance, VAE, or text encoder
