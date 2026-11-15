"""Sieve MLE for density-dependent McKean-Vlasov diffusions, arXiv:2609.01166.

The model is eq (1.1),

    dX_t = -Xi(p_t(X_t)) grad V(X_t) dt + sqrt(2) dW_t,

with a known confining potential V and an unknown interaction function
Xi: R+ -> R+.  Everything here works with the zero-flux stationary density of
Proposition 3.4 and the one-dimensional likelihood it induces, eq (3.9).

float64 is on.  The forward map is an ODE plus a root find plus a log
likelihood, and at float32 the normalisation root find is worth about four
digits, which is not enough.  On NVIDIA the default TF32 matmuls would make
that worse.
"""

import jax

jax.config.update("jax_enable_x64", True)

from mkv_sieve.potential import OUPotential  # noqa: E402
from mkv_sieve.profile import (  # noqa: E402
    SolverConfig,
    solve_profile,
    profile_log_q,
    profile_a,
    profile_log_pi,
    profile_energy_cdf,
)
from mkv_sieve.sieve import (  # noqa: E402
    requ,
    clamp_tau,
    init_requ_params,
    requ_net,
    make_xi,
)
from mkv_sieve.sample import sample_energies, sample_particles  # noqa: E402
from mkv_sieve.mle import (  # noqa: E402
    nll,
    fit_mle,
    ise_on_interval,
    l2_on_interval,
    kl_stationary,
)

__all__ = [
    "OUPotential",
    "SolverConfig",
    "solve_profile",
    "profile_log_q",
    "profile_a",
    "profile_log_pi",
    "profile_energy_cdf",
    "requ",
    "clamp_tau",
    "init_requ_params",
    "requ_net",
    "make_xi",
    "sample_energies",
    "sample_particles",
    "nll",
    "fit_mle",
    "ise_on_interval",
    "l2_on_interval",
    "kl_stationary",
]
