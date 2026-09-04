# jax-head-complexity

JAX implementation of "The Head Complexity of Boolean Functions in Single-Layer Attention" (arXiv:2609.04046).

## install

uv pip install -e .

## run

python -m head_complexity --bits 4

Theorem 2 construction: k heads compute k-bit parity. Also the alternating-sum obstruction of Theorem 1 and the monomial-vote construction of Theorem 5. Not a paper-for-paper check of the counting lower bound of Theorem 7, which is non-constructive.
