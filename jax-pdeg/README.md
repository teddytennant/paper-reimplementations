# jax-pdeg

JAX implementation of the single-loop projected damped extragradient method for smooth nonconvex-concave minimax problems.

https://arxiv.org/abs/2609.17973

## Install
pip install -e ".[tests]"

## Run
python -m pytest -q
python scripts/smoke.py

CPU only.

## What does not match the paper
No complexity lower-bound experiments, no distributionally robust / ATK training, no Moreau-envelope numerical evaluation of Φ_λ. Toy problems only. Parameter λ defaults to 2L as in the theorems.

Additional implementation notes:
- Warm-up length uses \(T_w=\lceil\log q/\log(1+128\beta)\rceil\) as in (4.31) and the contraction (C17); not \(\log(1+\beta/128)\).
- \(\alpha=(1+h\sqrt{2L\mu_y}/16)^{-1}\) so that \(\alpha\) is consistent with the \((1-\alpha)\gamma\) identity in (4.3).
- Euclidean simplex projection (Duchi sorting) is provided but not used in the paper's complexity theorems.
- The Moreau envelope \(\nabla\Phi_\lambda\) is not implemented; only \(\nabla p(z)=\lambda(z-x^\star(z))\) is referenced in comments.
