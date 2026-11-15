import jax.numpy as jnp
import numpy as np

from pointlam.grid import cell_indices, neighbor_mask


def test_same_cell_are_neighbors():
    cell_size = 1.0
    points = jnp.array(
        [
            [0.1, 0.2, 0.3],
            [0.8, 0.1, 0.4],
            [0.2, 0.9, 0.7],
        ],
        dtype=jnp.float32,
    )
    cells = cell_indices(points, cell_size)
    np.testing.assert_array_equal(cells, np.zeros((3, 3), dtype=np.int32))
    mask = neighbor_mask(points, cell_size)
    assert bool(jnp.all(mask))


def test_far_apart_points_are_not_neighbors():
    cell_size = 1.0
    points = jnp.array(
        [
            [0.2, 0.2, 0.2],
            [0.3, 0.1, 0.4],
            [10.2, 10.2, 10.2],
            [10.4, 10.1, 10.5],
        ],
        dtype=jnp.float32,
    )
    mask = np.array(neighbor_mask(points, cell_size))
    assert mask[0, 1] and mask[1, 0]
    assert mask[2, 3] and mask[3, 2]
    assert not mask[0, 2]
    assert not mask[0, 3]
    assert not mask[1, 2]
    assert not mask[1, 3]
    assert mask[0, 0] and mask[2, 2]


def test_adjacent_cells_are_neighbors():
    cell_size = 1.0
    points = jnp.array(
        [
            [0.5, 0.5, 0.5],
            [1.5, 0.5, 0.5],
        ],
        dtype=jnp.float32,
    )
    mask = neighbor_mask(points, cell_size)
    assert bool(mask[0, 1]) and bool(mask[1, 0])
