# intrinsic-atlas

JAX kernel of the posterior-mean DDPM iteration that recovers a population template (intrinsic atlas) from a diffusion reverse process, following arXiv:2609.30566.

## Install

```
python3 -m venv .venv
.venv/bin/pip install -e ".[tests]"
```

## Run

```
.venv/bin/python -m pytest
.venv/bin/python train.py
```

## What does not match the paper

Equation (2) is typeset ambiguously in the PDF. This package uses the standard VP-SDE/DDPM Tweedie formula \(\hat x_0=(x_t+(1-\bar\alpha_t)\nabla\log f)/\sqrt{\bar\alpha_t}\), which recovers \(E[x_0\mid x_t]\) for a Gaussian and matches the affine denoiser (eq. 3). The linear \(\beta\) schedule is Ho et al. (2020), \(\beta\in[10^{-4},0.02]\). There are no pretrained medical generators, no registration metrics, and no large-image experiments. `train.py` is a tiny 2D MLP DDPM and is not required to match Table 2 numbers. Atlas readout uses \(t=0\) rather than the paper's operating point \(t^*=99\).
