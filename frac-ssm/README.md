JAX kernel of FRAC fractional state-space layers (arXiv:2609.36314).

## Install

```
python3 -m venv .venv
.venv/bin/pip install -e ".[tests]"
```

## Run

```
.venv/bin/python -m pytest -q
.venv/bin/python train.py
```

## What does not match the paper

No 1.3B language model, no Triton kernels, no Mamba/GDN baselines, and no LM numbers. Widths are tiny (d=16, M=4, H=2). Algorithm 1 uses β̃=1-ρ̃ rather than (1-ρ̃)/λ; that is the paper's implemented form (Remark 7). Exact ZOH (1-ρ)/λ is tested separately.
