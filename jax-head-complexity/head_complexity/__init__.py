"""Constructions from arXiv:2609.04046, "The Head Complexity of Boolean Functions
in Single-Layer Attention" (Rajaraman, Sundaram, Tesfaye).

Modules follow the paper: `layer` is Section 2 (Definitions 1 and 2, the scalar
normal form, the decision polynomial F_x, Theorem 8), `parity` is Theorem 2 and
Lemma 4, `monomial` is Theorem 5 and Lemma 2, `obstruction` is the alternating
sum of Theorem 1.

float64 goes on at import time and stays on. Head j of Theorem 2 has denominator
c + j on Hamming weight c, the parity margin min_c |P(c)/D(c)| is 1.1e-3 already
at k = 4, and Theorem 1's identity is an exact cancellation. None of that
survives float32, and still less the TF32 matmuls a GPU picks by default.
"""

import jax as _jax

_jax.config.update("jax_enable_x64", True)

__all__ = ["layer", "monomial", "obstruction", "parity"]
