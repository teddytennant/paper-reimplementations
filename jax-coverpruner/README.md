# jax-coverpruner

JAX implementation of "Who Speaks for the Pruned? Visual Token Pruning as Coverage Optimization" (arXiv:2609.03158).

## install

uv pip install -e .

## run

python -m coverpruner.demo --tokens 576 --budget 64

The selector is Algorithm 1 on arrays: calibrated projector-space coverage (Eqs. 2-3) and greedy demand-weighted RCM (Eq. 6). There is no VLM in the loop, so this is not a paper-for-paper reproduction of the LLaVA tables.
