"""Turn on double precision and exact matmuls, then check it actually happened.

Every exactness claim in the paper is an equality: detailed balance (Eq. 9),
the marginal-acceptance ratio identity in the proof of Theorem 2, and
pi @ P = pi. Checking those at float32 is pointless, and on NVIDIA a float32
matmul silently runs at TF32 and loses about 1e-2 of relative precision, which
would swamp the thing being measured. So we ask for x64 and for the highest
matmul precision, and raise if either request was ignored.
"""

import jax
import jax.numpy as jnp


def enable() -> None:
    """Enable float64 and highest-precision matmuls. Raises if x64 did not take.

    Call this before any jax array is created. JAX only honours
    ``jax_enable_x64`` at trace time, so flipping it after a float32 array has
    been built leaves that array at float32.
    """
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "highest")
    if jnp.zeros(1, dtype=jnp.float64).dtype != jnp.dtype("float64"):
        raise RuntimeError("jax_enable_x64 did not take effect")


def is_x64() -> bool:
    """True if float64 arrays survive as float64."""
    return jnp.zeros(1, dtype=jnp.float64).dtype == jnp.dtype("float64")
