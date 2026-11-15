import jax

# Distances, BP messages and Jacobians go through products of tanh. Float32
# is enough for training but not for the closed-form checks, so tests run in
# float64. Training itself stays float32.
jax.config.update("jax_enable_x64", True)
