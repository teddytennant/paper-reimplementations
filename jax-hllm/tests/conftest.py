import jax

# Finite difference gradient checks and the doubly stochastic tolerances below
# are not meaningful in float32.
jax.config.update("jax_enable_x64", True)
