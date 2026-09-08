# jax-ct-idci

JAX implementation of copula-transformed iterative data-consistent inversion
from "Copula Transformations for Data-Consistent Inversion" (arXiv:2609.02832).

## install

uv pip install -e .

## run

python -m pytest -q

Reproduces the linear-Gaussian example of section 6.1.1 at modest sample size.
Not a reproduction of the trommel-screen experiment or the KDE non-Gaussian case.
