"""Every test runs in float64.

Two reasons. The moment and mass balance identities checked here hold to roundoff
in the discretisation, so float32 would hide a real bug behind 1e-6 noise. And
float32 matmuls on NVIDIA default to TF32, which would put about 1e-2 relative
error into anything that goes through the MLP in hybrid.py.
"""

import jax

jax.config.update("jax_enable_x64", True)
