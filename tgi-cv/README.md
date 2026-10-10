Small JAX package for transformed Gaussian incremental random-walk Metropolis with probabilists Hermite Poisson control variates.

Install from this directory with `pip install -e .` (jax, jaxlib, optax, pytest). Tests also run against `src` via pytest pythonpath.

Run the tests with `JAX_PLATFORMS=cpu python -m pytest tests -q`. The GPU smoke script is `python smoke_gpu.py`; the same script is the CPU check under `JAX_PLATFORMS=cpu`.

This is not the paper. There is no logistic regression, no eight schools, no TESS, no TGI-MALA, and no paper-scale ESS.
