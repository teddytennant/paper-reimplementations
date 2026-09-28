Toy JAX kernel of the alternating-denoising split-Gibbs sampler from [arXiv:2609.30539](https://arxiv.org/abs/2609.30539), with an unfolded likelihood denoiser on small real phase retrieval \(y=|Ax|+\varepsilon\).

## Install

```
python -m venv .venv
.venv/bin/pip install -e ".[tests]"
```

## Run

```
.venv/bin/pytest
.venv/bin/python train.py
```

## What does not match the paper

The paper's experiments use FFHQ \(256\times 256\) coded-diffraction phase retrieval \(Y=|\mathrm{FFT}(B\odot X)|+\varepsilon\), a pretrained image diffusion prior, six unfolded layers with learned forward/backward masks on the physical mask support, a four-layer proximal MLP, and a 22-iteration MCMC runtime comparison. This kernel uses \(n=8\) real vectors, a closed-form Gaussian prior denoiser, two unfold layers, a three-layer coordinate-wise MLP, and a known dense \(A\) with no learned masks. Paper eq. (6) leaves \(\sigma(\cdot)\) user-specified; the VP identity \(\sigma^2(t)=\exp(\int_0^t\beta)-1\) is one schedule used here, not a displayed formula. No ImageNet checkpoints, cameras, or PSNR/SSIM tables.
