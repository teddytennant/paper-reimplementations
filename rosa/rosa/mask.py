"""Bottom-rho mask by absolute weight, smallest |w| first."""

from __future__ import annotations

import math

import numpy as np


def mask_count(n: int, rho: float) -> int:
    """Number of parameters kept by a bottom-rho mask.

    k = floor(rho * n). If rho is positive and that product rounds down to
    zero, one parameter is kept so a nonzero density still trains something.
    """

    if n < 0:
        raise ValueError("n must be non-negative")
    if rho < 0 or rho > 1:
        raise ValueError("rho must lie in [0, 1]")
    if n == 0 or rho == 0:
        return 0
    k = int(math.floor(float(rho) * int(n) + 1e-9))
    if k < 1:
        k = 1
    return min(k, int(n))


def select_bottom_rho(values, rho: float) -> np.ndarray:
    """Boolean mask of the bottom-rho entries by absolute value.

    Ties break toward the smaller flat index. The mask marks the smallest
    |w|, not the largest.
    """

    array = np.asarray(values)
    flat = np.abs(np.asarray(array, dtype=np.float64).reshape(-1))
    chosen = np.zeros(flat.size, dtype=bool)
    k = mask_count(flat.size, rho)
    if k == 0:
        return chosen.reshape(array.shape)
    order = np.lexsort((np.arange(flat.size), flat))
    chosen[order[:k]] = True
    return chosen.reshape(array.shape)


def active_parameter_mask(params, active_ids, rho: float):
    """Bottom-rho slots of S_e, as (layer_id, param_name, flat_index).

    Slots are visited in layer-id order, weight then bias, so ties are stable.
    """

    records = []
    for lid in sorted(active_ids, key=_layer_sort_key):
        layer = params[lid]
        for name in ("w", "b"):
            flat = np.asarray(layer[name]).reshape(-1)
            for index, value in enumerate(flat):
                records.append((lid, name, int(index), float(value)))
    if not records:
        return ()
    values = np.array([row[3] for row in records], dtype=np.float64)
    chosen = select_bottom_rho(values, rho).reshape(-1)
    return tuple(records[i][:3] for i in range(len(records)) if bool(chosen[i]))


def _layer_sort_key(layer_id):
    if isinstance(layer_id, (int, np.integer)):
        return (0, int(layer_id), "")
    return (1, 0, str(layer_id))
