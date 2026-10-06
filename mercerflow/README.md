MercerFlow, flow matching in the Mercer eigenbasis of a GP prior, from https://arxiv.org/abs/2610.06039.

```bash
JAX_PLATFORMS=cpu python -m pytest -q
```

Tests check the OU Gram (equation 1), the past/future partition and conditional moments (equations 2 and 3), orthonormal codecs and the encode/decode maps (equation 4), the Mercer expansion, descending spectrum, sign convention, and Cov(c) = Lambda (equation 5), a hand-built Fourier column (equation 6) and DCT column (equation 7), PCA centering only inside the sample covariance (equation 8), the linear interpolant and the isometry ||u|| = ||yf - y0|| (equations 9 and 10), the post-linear AdaLN MLP with zero-initialised gamma and beta (equation 27), one finite CPU training step, Euler forecast shape, and the Proposition 2 Jacobi condition number on a small grid. They do not match paper CRPS, do not use GluonTS or the benchmark series, and run at small L rather than 384. Hidden width in the tests may be smaller than 64. The PDF names RBF and periodic kernels but does not state a periodic formula, so only the OU kernel of equation 1 is implemented. A 1x H200 smoke (200 steps, L=64) was cancelled while still pending; Slurm's start was 2026-10-08. The same script on CPU finished in 11.3s, with mean loss 59.07 over the first 20 steps and 36.94 over the last 20.
