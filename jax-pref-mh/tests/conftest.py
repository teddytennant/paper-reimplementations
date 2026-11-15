"""Turn on float64 before any test module builds a jax array.

pytest imports conftest first, so this runs before ``import jax.numpy``
anywhere else in the suite. The exactness tests compare quantities at 1e-12,
which float32 cannot represent, and on NVIDIA a float32 matmul defaults to
TF32 and would lose about 1e-2 of relative precision in the ``pi @ P`` check.
"""

from pref_mh.x64 import enable

enable()
