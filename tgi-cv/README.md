Small JAX package for transformed Gaussian incremental random-walk Metropolis with probabilists Hermite Poisson control variates.

Install from this directory with `/home/nixos/arxiv-impl-work/bin/py -m pip install -e .` (or any environment that already has jax, jaxlib, optax, and pytest; tests also run against `src` via pytest pythonpath).

Run the tests with `JAX_PLATFORMS=cpu /home/nixos/arxiv-impl-work/bin/py -m pytest tests -q`. The GPU smoke script is `python smoke_gpu.py`; the same script is the CPU check under `JAX_PLATFORMS=cpu`.

This is not the paper. There is no logistic regression, no eight schools, no TESS, no TGI-MALA, and no paper-scale ESS.
