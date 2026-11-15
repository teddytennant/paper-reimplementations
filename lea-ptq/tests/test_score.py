"""Hand checks for the local error score."""

import math

import jax.numpy as jnp
import pytest

from lea_ptq.score import local_score


def _sample(second):
    # bits=3, qmax=3, row max=1, scale=1/3.
    # second=0.5 -> ratio 1.5 -> jnp.round = 2 -> code*scale = 2/3.
    # error on x=[0, 1] has l2 norm |0.5 - 2/3| = 1/6.
    # second=0.25 -> ratio 0.75 -> round = 1 -> code*scale = 1/3.
    # error l2 norm |0.25 - 1/3| = 1/12.
    # A second layer with a single max entry quantizes exactly and adds 0.
    return (
        (jnp.asarray([[1.0, second]]), jnp.asarray([0.0, 1.0])),
        (jnp.asarray([[1.0]]), jnp.asarray([1.0])),
    )


def test_two_samples_match_hand_statistics():
    result = local_score([_sample(0.5), _sample(0.25)], bits=3, delta=0.1)
    # P = [1/6, 1/12], mu = 1/8.
    # sample variance = 2 * (1/24)**2, sigma = sqrt(2)/24.
    # sqrt((1-0.1)/0.1) = 3, so s_loc = 1/8 + sqrt(2)/8.
    mu = 0.125
    sigma = math.sqrt(2.0) / 24.0
    s_loc = mu + 3.0 * sigma
    assert float(result["mu"]) == pytest.approx(mu)
    assert float(result["sigma"]) == pytest.approx(sigma)
    assert float(result["s_loc"]) == pytest.approx(s_loc)
    assert result["s_loc"] == pytest.approx(mu + sigma * math.sqrt(0.9 / 0.1))


def test_single_sample_raises():
    with pytest.raises(ValueError):
        local_score([_sample(0.5)], bits=3, delta=0.1)


def test_delta_outside_unit_interval_raises():
    samples = [_sample(0.5), _sample(0.25)]
    with pytest.raises(ValueError):
        local_score(samples, bits=3, delta=0.0)
    with pytest.raises(ValueError):
        local_score(samples, bits=3, delta=1.0)
