JAX kernel of MEFPIA (arXiv:2609.21944) for N-player EGW quantum games.

## Install

From this directory, with the repo venv:

    source venv/bin/activate
    pip install -e ".[test]"

If the venv is already populated with jax, numpy, and pytest, adding `src` to `PYTHONPATH` is enough.

## Tests

    python -m pytest -q

CPU only.

## What does not match the paper

Only geometric cooling F(k)=alpha^k is implemented. There is no T2V construction, no mini-experiments 2 or 3, no Experiment 2, and no 100-random-init sweep plots. MMWU is a small Kale-style matrix update for comparison, not a full reproduction of Lotidis et al.
