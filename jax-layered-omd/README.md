# jax-layered-omd

JAX implementation of "Independent Reinforcement Learning in Discounted Markov Games"
(arXiv:2609.00504).

## install

uv pip install -e .

## run

python -m layered_omd.experiments --episodes 40 --blocks 8 --block-length 16

Layered optimistic mirror descent (Algorithms 1 and 2) on the paper's toy Markov games. Not a paper-for-paper reproduction of the T=1000 / K=1000 CCE-gap plots; the hardness result is not implemented.
