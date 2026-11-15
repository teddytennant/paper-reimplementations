JAX kernel of MEFPIA (arXiv:2609.21944) for N-player EGW quantum games.

## Install

    pip install -e ".[test]"

## Tests

    python -m pytest -q

CPU only.

## What does not match the paper

Only geometric cooling F(k)=alpha^k is implemented. Mini-experiments 2 and 3, Experiment 2, and the 100-random-init sweep plots are not included. MMWU is a small Kale-style matrix update for comparison, not a full reproduction of Lotidis et al.
