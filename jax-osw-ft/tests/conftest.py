"""Everything runs in float64.

Several tests compare against exact rational arithmetic or take finite
differences. On NVIDIA, XLA runs float32 matmuls at TF32 by default, so a
float32 version of those tests would pass here and fail on an H200 for reasons
that have nothing to do with the method.
"""

import jax

jax.config.update("jax_enable_x64", True)
