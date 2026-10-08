"""Tensor-train SVD for an interleaved weight tensor.

Cores G^{(k)} have shape (r_{k-1}, m_k, n_k, r_k), with r_0 = r_K = 1.
The input tensor is already in eq 1 order: (m1, n1, m2, n2, ..., mK, nK).
TTN is not implemented.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def interleave(matrix, m_factors, n_factors):
    """Reshape (m, n) to eq 1 order (m1, n1, ..., mK, nK).

    C-order reshape to (m1, ..., mK, n1, ..., nK), then transpose so each
    output mode sits next to its input mode.
    """
    m_factors = tuple(int(v) for v in m_factors)
    n_factors = tuple(int(v) for v in n_factors)
    if len(m_factors) != len(n_factors):
        raise ValueError("m_factors and n_factors must have the same length")
    tensor = jnp.reshape(matrix, m_factors + n_factors)
    k = len(m_factors)
    perm = []
    for i in range(k):
        perm.append(i)
        perm.append(k + i)
    return jnp.transpose(tensor, perm)


def _bond_ranks(ranks, n_bonds):
    if ranks is None:
        return [None] * n_bonds
    if isinstance(ranks, bool) or isinstance(ranks, int):
        if isinstance(ranks, bool):
            raise TypeError("ranks must be an int or a sequence of ints")
        return [int(ranks)] * n_bonds
    rank_list = [int(r) for r in list(ranks)]
    if len(rank_list) != n_bonds:
        raise ValueError(
            f"ranks must have length {n_bonds} (one bond rank per split), got {len(rank_list)}"
        )
    return rank_list


def tt_svd(tensor_interleaved, ranks=None):
    """TT-SVD of an interleaved tensor.

    For k = 1..K-1, matricize to (r_prev * m_k * n_k, -1), take a thin SVD,
    keep r_k singular values (clamped to the SVD dimension), store U as the
    core (r_prev, m_k, n_k, r_k), and continue with S @ Vh.

    ranks is None for full rank (every singular value), an int applied to
    every bond, or a sequence of length K-1 giving r_1, ..., r_{K-1}.
    """
    # Float64 SVD so full-rank reconstruction stays inside 1e-4.
    tensor = jnp.asarray(tensor_interleaved, dtype=jnp.float64)
    shape = tuple(int(s) for s in tensor.shape)
    if len(shape) == 0 or len(shape) % 2 != 0:
        raise ValueError(
            "tensor_interleaved must have shape (m1, n1, ..., mK, nK)"
        )
    k_cores = len(shape) // 2
    modes = [(shape[2 * i], shape[2 * i + 1]) for i in range(k_cores)]
    rank_list = _bond_ranks(ranks, k_cores - 1)

    cores = []
    current = tensor
    r_prev = 1
    for k in range(k_cores - 1):
        m_k, n_k = modes[k]
        left = r_prev * m_k * n_k
        matrix = jnp.reshape(current, (left, -1))
        u, s, vh = jnp.linalg.svd(matrix, full_matrices=False)
        max_rank = int(s.shape[0])
        if rank_list[k] is None:
            r_k = max_rank
        else:
            r_k = min(int(rank_list[k]), max_rank)
        if r_k < 1:
            raise ValueError("TT rank must be at least 1")
        u = u[:, :r_k]
        s = s[:r_k]
        vh = vh[:r_k, :]
        cores.append(jnp.reshape(u, (r_prev, m_k, n_k, r_k)))
        rest = []
        for j in range(k + 1, k_cores):
            rest.extend(modes[j])
        # S @ Vh, absorbed into the remaining tensor (not into U).
        current = jnp.reshape(s[:, None] * vh, (r_k, *rest))
        r_prev = r_k

    m_last, n_last = modes[-1]
    if k_cores == 1:
        last = jnp.reshape(tensor, (1, m_last, n_last, 1))
    else:
        last = jnp.reshape(current, (r_prev, m_last, n_last, 1))
    cores.append(last)
    return cores


def _contract_bond(left, right):
    bond = left.shape[-1]
    if right.shape[0] != bond:
        raise ValueError("TT bond dimensions do not match")
    flat = jnp.matmul(jnp.reshape(left, (-1, bond)), jnp.reshape(right, (bond, -1)))
    return jnp.reshape(flat, left.shape[:-1] + right.shape[1:])


def densify(cores):
    """Contract cores to the dense matrix (m, n).

    Undoes eq 1 interleaving: (m1, n1, ..., mK, nK) goes back to
    (m1, ..., mK, n1, ..., nK) and then to (m, n) in C order.
    """
    if not cores:
        raise ValueError("cores must be non-empty")
    acc = jnp.asarray(cores[0])
    for core in cores[1:]:
        acc = _contract_bond(acc, jnp.asarray(core))
    if acc.shape[0] != 1 or acc.shape[-1] != 1:
        raise ValueError("boundary TT ranks must be 1")
    acc = jnp.reshape(acc, acc.shape[1:-1])
    k_cores = len(cores)
    m_axes = list(range(0, 2 * k_cores, 2))
    n_axes = list(range(1, 2 * k_cores, 2))
    acc = jnp.transpose(acc, m_axes + n_axes)
    m = 1
    n = 1
    for core in cores:
        m *= int(core.shape[1])
        n *= int(core.shape[2])
    return jnp.reshape(acc, (m, n))


def tt_matvec(cores, x):
    """Contract TT cores with a vector x of shape (n,).

    Returns y of shape (m,). Does not materialize the dense matrix.
    Input modes are consumed in C order, n1 slowest, matching eq 1.
    """
    if not cores:
        raise ValueError("cores must be non-empty")
    x = jnp.asarray(x)
    if x.ndim != 1:
        raise ValueError("x must be a vector of shape (n,)")
    n_factors = [int(core.shape[2]) for core in cores]
    n = 1
    for factor in n_factors:
        n *= factor
    if int(x.shape[0]) != n:
        raise ValueError(f"x has length {int(x.shape[0])}, expected {n}")

    # Running state: (m_done, r_prev, n_remaining).
    m_done = 1
    res = jnp.reshape(x, (m_done, 1, n))
    for core in cores:
        core = jnp.asarray(core)
        r_in, m_k, n_k, r_out = (int(core.shape[0]), int(core.shape[1]), int(core.shape[2]), int(core.shape[3]))
        n_rem = int(res.shape[-1])
        if n_k == 0 or n_rem % n_k != 0:
            raise ValueError("input factor does not divide the remaining vector")
        n_rest = n_rem // n_k
        res = jnp.reshape(res, (m_done, r_in, n_k, n_rest))
        # res: (m_done, r_in, n_k, n_rest), core: (r_in, m_k, n_k, r_out)
        res = jnp.einsum("abcd,becf->aefd", res, core)
        m_done = m_done * m_k
        res = jnp.reshape(res, (m_done, r_out, n_rest))
    return jnp.reshape(res, (m_done,))
