"""Small stacks and inputs shared by the tests. Everything stays tiny on purpose."""

import jax
import jax.numpy as jnp

from gale import Conv, GlobalGate, MaxPool, Relu

F64 = jnp.float64


def conv(key, cin, cout, k=3, stride=1):
    k1, k2 = jax.random.split(key)
    scale = (k * k * cin) ** -0.5
    return Conv(
        w=jax.random.normal(k1, (k, k, cin, cout), dtype=F64) * scale,
        b=jax.random.normal(k2, (cout,), dtype=F64) * 0.1,
        stride=stride,
    )


def linear_stack(seed=0):
    """Convolutions only, no nonlinearity, so influence can be read off a gradient."""
    ks = jax.random.split(jax.random.PRNGKey(seed), 3)
    return [conv(ks[0], 2, 3, 3, 1), conv(ks[1], 3, 3, 3, 2), conv(ks[2], 3, 2, 5, 1)]


def conv_stack(seed=0):
    ks = jax.random.split(jax.random.PRNGKey(seed), 3)
    return [
        conv(ks[0], 2, 4, 3, 1),
        Relu(),
        conv(ks[1], 4, 4, 3, 2),
        Relu(),
        conv(ks[2], 4, 3, 3, 1),
    ]


def pool_stack(seed=0):
    ks = jax.random.split(jax.random.PRNGKey(seed), 2)
    return [conv(ks[0], 2, 4, 3, 1), Relu(), MaxPool(k=3, stride=2), conv(ks[1], 4, 3, 3, 1)]


def gate_stack(seed=0):
    """Contains a squeeze-and-excite style global reduction, so PPBI cannot tile it."""
    ks = jax.random.split(jax.random.PRNGKey(seed), 3)
    return [
        conv(ks[0], 2, 4, 3, 1),
        Relu(),
        GlobalGate(
            gain=jnp.full((4,), 2.0, F64), bias=jax.random.normal(ks[2], (4,), dtype=F64)
        ),
        conv(ks[1], 4, 3, 3, 2),
    ]


def wide_stack(seed=0):
    """Channel expansion at full resolution, the shape that makes early activations
    the memory bottleneck in an inverted residual network."""
    ks = jax.random.split(jax.random.PRNGKey(seed), 3)
    return [
        conv(ks[0], 1, 16, 3, 1),
        Relu(),
        conv(ks[1], 16, 16, 3, 2),
        Relu(),
        conv(ks[2], 16, 8, 3, 1),
    ]


def noise(seed, shape=(1, 32, 12, 2)):
    return jax.random.normal(jax.random.PRNGKey(seed), shape, dtype=F64)


def mixed_input(seed, h=48, w=12, c=3, high=1.0, low=3.0):
    """Smooth low frequency structure plus row-alternating detail.

    G_A keeps the first and loses the second, a sliced L_E pass keeps the second
    and gets the first wrong at its band edges, which is the regime Appendix C's
    alpha sweep lives in.
    """
    rows = jnp.arange(h, dtype=F64)[None, :, None, None]
    smooth = low * jnp.sin(2 * jnp.pi * rows / h)
    detail = high * ((-1.0) ** jnp.arange(h, dtype=F64))[None, :, None, None]
    jitter = jax.random.normal(jax.random.PRNGKey(seed), (1, h, w, c), dtype=F64)
    return smooth + detail + 0.3 * jitter
