"""Turn on float64 before any test module builds an array.

Two of the assertions in this suite are tight: Parseval to 1e-11 relative, and
the focal-exponent-zero reduction to the same. Both would fail on an NVIDIA GPU
in default float32, where matmuls fall back to TF32 and the FFT is less accurate
than the CPU one. x64 is a global flag, so setting it in conftest makes it
independent of which test module imports first. The training tests pass an
explicit float32 dtype through, so they keep running at single precision.
"""

import jax

jax.config.update("jax_enable_x64", True)
