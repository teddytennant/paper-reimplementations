"""MEFPIA kernel for EGW quantum games (arXiv:2609.21944)."""

from mefpia.ops import (
    hermitize,
    is_density,
    is_hermitian,
    random_density,
    von_neumann_entropy,
)
from mefpia.payoff import (
    as_matrix,
    as_tensor,
    effective_hamiltonian,
    payoff,
    payoffs,
)
from mefpia.algo import gibbs_state, mefpia, mmwu_step
from mefpia.pd import prisoners_dilemma, relative_payoff_error

__all__ = [
    "as_matrix",
    "as_tensor",
    "effective_hamiltonian",
    "gibbs_state",
    "hermitize",
    "is_density",
    "is_hermitian",
    "mefpia",
    "mmwu_step",
    "payoff",
    "payoffs",
    "prisoners_dilemma",
    "random_density",
    "relative_payoff_error",
    "von_neumann_entropy",
]
