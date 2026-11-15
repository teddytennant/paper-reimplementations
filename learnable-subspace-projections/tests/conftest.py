import jax

# Projector and merge checks use a 1e-5 absolute tolerance.
jax.config.update("jax_enable_x64", True)
