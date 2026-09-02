import jax
import jax.numpy as jnp
import pytest

from cachebridge import support


@pytest.mark.parametrize("lt,ht,k,hs,ds,dt", [(40, 8, 20, 8, 128, 128),
                                              (64, 8, 8, 8, 128, 128),
                                              (12, 4, 3, 5, 64, 96)])
def test_coefficient_ratio_is_exactly_hs(lt, ht, k, hs, ds, dt):
    """Eqs 12-13: C_full / C_local = Hs, independent of Ht and k."""
    c_full = support.coefficient_count(lt, ht, k, hs, ds, dt)
    c_local = support.coefficient_count(lt, ht, k, 1, ds, dt)
    assert c_full == c_local * hs
    assert c_local == 2 * lt * ht * (k * ds) * dt


def test_qwen3_input_widths_from_section_4_2():
    """Section 4.2 quotes 8,192 versus 1,024 per target head for Qwen3 14B -> 32B."""
    k, hs, ds = 8, 8, 128
    assert k * hs * ds == 8192
    assert k * ds == 1024


def test_head_local_is_the_assigned_slice_of_full_head():
    """Eq 11 selects exactly the a(h) blocks that eq 7 concatenates."""
    ls, n, hs, ds, ht = 5, 17, 4, 6, 4
    src = jax.random.normal(jax.random.PRNGKey(0), (ls, n, hs, ds))
    layers = jnp.array([1, 3])
    assign = support.cyclic_assignment(ht, 2)
    full = support.full_head_features(src, layers)
    local = support.head_local_features(src, layers, assign)
    assert full.shape == (n, len(layers) * hs * ds)
    assert local.shape == (ht, n, len(layers) * ds)
    for h in range(ht):
        g = int(assign[h])
        for slot in range(len(layers)):
            lo = (slot * hs + g) * ds
            assert jnp.array_equal(full[:, lo:lo + ds],
                                   local[h][:, slot * ds:(slot + 1) * ds])


def test_full_head_features_are_the_raw_source_blocks():
    src = jax.random.normal(jax.random.PRNGKey(1), (3, 9, 2, 4))
    layers = jnp.array([0, 2])
    full = support.full_head_features(src, layers)
    assert jnp.array_equal(full[:, :4], src[0, :, 0, :])
    assert jnp.array_equal(full[:, 4:8], src[0, :, 1, :])
    assert jnp.array_equal(full[:, 8:12], src[2, :, 0, :])


def test_assignments():
    assert jnp.array_equal(support.identity_assignment(8), support.cyclic_assignment(8, 0))
    assert jnp.array_equal(support.cyclic_assignment(4, 3), jnp.array([3, 0, 1, 2]))
    assert jnp.array_equal(support.group_assignment(8, 8), jnp.arange(8))
    assert jnp.array_equal(support.group_assignment(4, 8), jnp.array([0, 0, 1, 1, 2, 2, 3, 3]))
