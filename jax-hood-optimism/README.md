# jax-hood-optimism

JAX implementation of "Constant regret in general games via higher-order optimism" (arXiv:2609.04113).

## install

uv pip install -e .

## run

python -m hood.run --players 2 --actions 2 --rounds 400 --game matching-pennies

Algorithm 1 (HOOD) with the lifted entropic choice map of eqs. 19-21 and the discounted (N+1)-th order predictor of eq. 22. Toy self-play only. Not a paper-for-paper check of the O(N^3 log^2 K) constant, which is a worst-case bound, not a measured number.
