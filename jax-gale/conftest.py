"""float64 everywhere.

The exactness tests compare a sliced pass against a whole-map pass through the
same convolutions. On NVIDIA, float32 matmuls default to TF32, which leaves
about 1e-2 of relative error and would turn a real assertion into a coin flip
that happens to pass on CPU. x64 plus lax.Precision.HIGHEST in the library keeps
those comparisons meaningful on a GPU.
"""

import jax

jax.config.update("jax_enable_x64", True)
