"""Run the whole suite in float64.

Two tests need it: the scipy KL comparison, which is checked to 1e-12, and the
central-difference gradient check, which has no usable signal at float32.
Enabling x64 here rather than inside those tests keeps it before the first array
is created, which is when JAX reads the flag.
"""

import jax

jax.config.update("jax_enable_x64", True)
