"""Global float64.

This suite is also run on an H200, where a float32 matmul defaults to TF32,
roughly 10 mantissa bits. Several of the assertions here are on quantities that
come out of a matmul and then get differenced or ratioed: the off-diagonal mass
in equation (4), the trace ``Tr(Y^T S)`` in equation (6), and the elementwise
comparison of an SS-ESOAP trajectory against ``optax.adam``. Those have no
tolerance that is both meaningful on CPU and survivable under TF32, so the whole
suite runs in float64 and the tolerances below are stated for float64.

``test_dtypes.py`` is the exception and builds float32 arrays explicitly, to
check the float32 path still runs and stays close.
"""

import jax

jax.config.update("jax_enable_x64", True)
