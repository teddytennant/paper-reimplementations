import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from schur_replay.gptq import damped_inverse_factor, gptq_quantize, initial_group_scales
from schur_replay.nvfp4 import (
    E2M1_LEVELS,
    e4m3_positive_codes,
    exportable_scales,
    gamma_from_amax,
    quantize,
    rho,
)
from schur_replay.schur import (
    commit_group,
    conditional_score,
    inverse_upper_factor,
    project_to_grid,
    propose_scales,
    replay_group,
    schur_complement,
    schur_from_factor,
    select_scales,
)


def _rel(a, b):
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    scale = jnp.maximum(jnp.max(jnp.abs(a)), jnp.max(jnp.abs(b)))
    scale = jnp.maximum(scale, 1e-30)
    return jnp.max(jnp.abs(a - b)) / scale


def _assert_rel(a, b, tol=1e-5):
    err = float(_rel(a, b))
    assert err <= tol, err


def _spd(key, n, jitter=0.5):
    raw = jax.random.normal(key, (n, n), dtype=jnp.float64)
    gram = raw.T @ raw + jitter * jnp.eye(n, dtype=jnp.float64)
    return 0.5 * (gram + gram.T)


def _reference_e4m3_positive():
    """Independent bit decoder for positive finite E4M3 codes."""
    values = []
    for bits in range(128):
        exp = (bits >> 3) & 0xF
        mant = bits & 0x7
        if exp == 15 and mant == 7:
            continue
        if exp == 0:
            value = mant * 2.0 ** (-9)
        else:
            value = (1.0 + mant / 8.0) * 2.0 ** (exp - 7)
        if value == 0.0:
            continue
        values.append(value)
    return values


def test_e4m3_codes_and_gamma():
    codes = e4m3_positive_codes(dtype=jnp.float64)
    reference = jnp.asarray(_reference_e4m3_positive(), dtype=jnp.float64)
    assert codes.shape == (126,)
    assert int(codes.shape[0]) == 126
    assert float(jnp.max(codes)) == 448.0
    assert float(jnp.min(codes)) == 2.0 ** (-9)
    assert bool(jnp.all(jnp.isfinite(codes)))
    assert bool(jnp.all(codes > 0))
    assert bool(jnp.all(codes == reference))
    # exp=15, mantissa=7 would be 480 and is the NaN encoding.
    assert not bool(jnp.any(codes == 480.0))
    assert not bool(jnp.any(codes == 0.0))

    amax = jnp.asarray(24.0, dtype=jnp.float64)
    gamma = gamma_from_amax(amax)
    assert float(gamma) == (448.0 * 6.0) / 24.0
    scales = exportable_scales(amax)
    assert scales.shape == (126,)
    assert float(jnp.max(scales)) == 448.0 / float(gamma)
    assert float(jnp.max(scales)) == 24.0 / 6.0
    assert bool(jnp.all(scales[1:] > scales[:-1]))


def test_round_trip_exact_level_and_ties_toward_zero():
    scale = 0.5
    for level in E2M1_LEVELS:
        for sign in (1.0, -1.0):
            weight = sign * scale * level
            reconstructed = float(quantize(weight, scale))
            assert reconstructed == weight
    # Midpoints: 0|0.5, 0.5|1, 1|1.5, 1.5|2, 2|3, 3|4, 4|6.
    assert float(rho(0.25)) == 0.0
    assert float(rho(-0.25)) == 0.0
    assert float(rho(0.75)) == 0.5
    assert float(rho(1.25)) == 1.0
    assert float(rho(-1.75)) == -1.5
    assert float(rho(2.5)) == 2.0
    assert float(rho(-3.5)) == -3.0
    assert float(rho(5.0)) == 4.0
    assert float(rho(-5.0)) == -4.0
    assert float(rho(5.0 + 1e-12)) == 6.0
    assert float(rho(100.0)) == 6.0
    assert float(rho(-100.0)) == -6.0
    # A midpoint input does not round-trip; the tie goes toward zero.
    midpoint = scale * 2.5
    assert float(quantize(midpoint, scale)) == scale * 2.0


def test_schur_complement_matches_factor():
    for seed, n, width in ((0, 12, 4), (1, 9, 3), (2, 16, 5)):
        gram = _spd(jax.random.PRNGKey(seed), n)
        factor = inverse_upper_factor(gram)
        _assert_rel(factor.T @ factor, jnp.linalg.inv(gram), tol=1e-5)
        # No prefix: leading group versus the future.
        from_blocks = schur_complement(gram, 0, width)
        from_factor = schur_from_factor(factor[:width, :width])
        _assert_rel(from_blocks, from_factor, tol=1e-5)
        # Prefix fixed: compare on the trailing conditional Hessian.
        prefix = 3
        if prefix + width < n:
            trailing = factor[prefix:, prefix:]
            conditional = jnp.linalg.inv(trailing.T @ trailing)
            conditional = 0.5 * (conditional + conditional.T)
            from_blocks = schur_complement(conditional, 0, width)
            from_factor = schur_from_factor(factor[prefix : prefix + width, prefix : prefix + width])
            _assert_rel(from_blocks, from_factor, tol=1e-5)


def test_one_column_score_equals_gptq_residual():
    """Section 4.1.2: one column has S_G = 1/u_ii^2 and score = e_i^2.

    e_i = (w_i - q_i) / u_ii is the GPTQ residual. The score is that residual
    squared. S_G = 1/H_ii does not hold for the original Hessian; the factor
    identity is the reduction proved in the paper.
    """
    gram = _spd(jax.random.PRNGKey(4), 7)
    factor = inverse_upper_factor(gram)
    column = 3
    diagonal = float(factor[column, column])
    metric = schur_from_factor(factor[column : column + 1, column : column + 1])
    _assert_rel(metric[0, 0], 1.0 / diagonal**2, tol=1e-5)
    # Leading column also matches the Schur complement of the original Hessian.
    leading = schur_complement(gram, 0, 1)
    _assert_rel(leading, schur_from_factor(factor[:1, :1]), tol=1e-5)
    _assert_rel(leading[0, 0], 1.0 / float(factor[0, 0]) ** 2, tol=1e-5)

    weight = 0.37
    scale = 0.2
    code = float(scale * rho(weight / scale))
    residual = (weight - code) / diagonal
    block = jnp.array([[weight]], dtype=jnp.float64)
    replayed = replay_group(
        block,
        factor[column : column + 1, column : column + 1],
        jnp.array([scale], dtype=jnp.float64),
    )
    score = conditional_score(replayed - block[:, None, :], metric)
    _assert_rel(score, residual**2, tol=1e-5)
    assert float(jnp.max(jnp.abs(replayed[0, 0, 0] - code))) <= 1e-12


def _sequential_replay(row, factor, scale):
    """Independent column loop. Uses the same rho, private Python state."""
    width = len(row)
    state = [float(row[j]) for j in range(width)]
    codes = []
    for j in range(width):
        code = float(scale) * float(rho(state[j] / float(scale)))
        error = (state[j] - code) / float(factor[j, j])
        codes.append(code)
        for k in range(j + 1, width):
            state[k] = state[k] - error * float(factor[j, k])
    return codes


def test_replay_is_private_and_matches_sequential():
    factor = jnp.array(
        [
            [1.2, 0.4, -0.3, 0.2],
            [0.0, 0.8, 0.5, -0.1],
            [0.0, 0.0, 1.5, 0.6],
            [0.0, 0.0, 0.0, 0.9],
        ],
        dtype=jnp.float64,
    )
    w0 = jnp.array(
        [
            [0.7, -0.4, 1.1, 0.2],
            [1.3, 0.5, -0.8, 0.9],
        ],
        dtype=jnp.float64,
    )
    scales = jnp.array(
        [
            [0.3, 1.1],
            [0.5, 0.2],
        ],
        dtype=jnp.float64,
    )
    snapshot = jnp.array(w0)
    batched = replay_group(w0, factor, scales)
    assert bool(jnp.all(w0 == snapshot))
    # Column 0 depends only on w0, so another candidate cannot leak into it.
    expected_first = scales * rho(w0[:, :1] / scales)
    _assert_rel(batched[:, :, 0], expected_first, tol=1e-12)
    assert not bool(jnp.allclose(batched[:, 0, :], batched[:, 1, :]))
    for row in range(w0.shape[0]):
        for cand in range(scales.shape[1]):
            alone = replay_group(w0[row : row + 1], factor, scales[row : row + 1, cand])
            _assert_rel(batched[row, cand], alone[0, 0], tol=1e-12)
            reference = _sequential_replay(w0[row], factor, scales[row, cand])
            _assert_rel(batched[row, cand], jnp.asarray(reference), tol=1e-10)


def test_fixed_code_update_matches_formula():
    metric = jnp.array(
        [
            [2.0, 0.2, 0.0, 0.1],
            [0.2, 1.5, -0.1, 0.0],
            [0.0, -0.1, 1.2, 0.3],
            [0.1, 0.0, 0.3, 1.8],
        ],
        dtype=jnp.float64,
    )
    w0 = jnp.array(
        [
            [1.0, 0.4, -1.2, 3.0],
            [0.2, -0.8, 2.5, -0.1],
        ],
        dtype=jnp.float64,
    )
    s0 = jnp.array([1.0, 0.5], dtype=jnp.float64)
    # Hand-checked codes for this toy. 5.0 is a midpoint and rounds toward zero.
    expected_codes = jnp.array(
        [
            [1.0, 0.5, -1.0, 3.0],
            [0.5, -1.5, 4.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    got_codes = rho(w0 / s0[:, None])
    assert bool(jnp.all(got_codes == expected_codes))
    expected = []
    for row in range(2):
        code = expected_codes[row]
        weight = w0[row]
        denom = code @ metric @ code
        numer = code @ metric @ weight
        assert float(denom) > 1e-12
        expected.append(numer / denom)
    expected = jnp.asarray(expected)
    updated = propose_scales(w0, metric, s0, iterations=1)
    _assert_rel(updated, expected, tol=1e-12)
    # A vanishing denominator keeps the incoming scale.
    tiny = jnp.full((1, 4), 1e-9, dtype=jnp.float64)
    kept = propose_scales(tiny, metric, jnp.array([1.25], dtype=jnp.float64), iterations=3)
    assert float(kept[0]) == 1.25


def test_selected_scale_does_not_increase_surrogate():
    layouts = ((8, 4, 4, 3), (16, 4, 8, 2))
    for case, (width, prefix, future, rows) in enumerate(layouts):
        n = prefix + width + future
        for seed in range(4):
            key = jax.random.PRNGKey(100 + 10 * case + seed)
            k_h, k_w, k_s = jax.random.split(key, 3)
            gram = _spd(k_h, n, jitter=0.8)
            factor = inverse_upper_factor(gram)
            block_factor = factor[prefix : prefix + width, prefix : prefix + width]
            metric = schur_from_factor(block_factor)
            w0 = jax.random.normal(k_w, (rows, width), dtype=jnp.float64)
            amax = jnp.max(jnp.abs(w0)) + 0.5
            grid = exportable_scales(amax)
            raw = jax.random.uniform(
                k_s,
                (rows,),
                dtype=jnp.float64,
                minval=float(grid[1]),
                maxval=float(grid[-2]),
            )
            raw = raw.at[0].set(grid[grid.shape[0] // 3])
            winner, info = select_scales(w0, block_factor, metric, raw, grid, half_width=4, iterations=3)
            projected = info["projected_old"]
            on_grid = jnp.any(jnp.abs(info["candidates"] - projected[:, None]) == 0, axis=1)
            assert bool(jnp.all(on_grid))
            q_win = replay_group(w0, block_factor, winner)
            q_old = replay_group(w0, block_factor, projected)
            loss_win = conditional_score(q_win - w0[:, None, :], metric)[:, 0]
            loss_old = conditional_score(q_old - w0[:, None, :], metric)[:, 0]
            assert bool(jnp.all(jnp.isfinite(loss_win)))
            assert bool(jnp.all(loss_win <= loss_old + 1e-8))
            assert bool(jnp.all(winner > 0))


def test_ties_prefer_old_then_smaller():
    grid = e4m3_positive_codes(dtype=jnp.float64)
    factor = jnp.array([[1.0]], dtype=jnp.float64)
    metric = jnp.array([[1.0]], dtype=jnp.float64)
    # All-zero row: every scale has loss 0, so the projected old scale wins.
    # Unit row: several grid scales reproduce 1 exactly; the old scale does not.
    w0 = jnp.array([[0.0], [1.0]], dtype=jnp.float64)
    s_old = jnp.array([3.0, 448.0], dtype=jnp.float64)
    winner, info = select_scales(
        w0, factor, metric, s_old, grid, half_width=126, iterations=1, tie_tol=1e-8
    )
    assert float(info["projected_old"][0]) == 3.0
    assert float(winner[0]) == 3.0
    assert float(grid[0]) < 3.0

    reproduced = grid[jnp.abs(grid * rho(1.0 / grid) - 1.0) == 0.0]
    assert int(reproduced.shape[0]) >= 2
    assert not bool(jnp.any(reproduced == 448.0))
    smallest = float(jnp.min(reproduced))
    assert smallest == 0.25
    assert float(winner[1]) == smallest
    old_loss = float((448.0 * rho(1.0 / 448.0) - 1.0) ** 2)
    assert old_loss > 1e-8

    # Off-grid old scale: the tie still prefers the projection, not a smaller scale.
    off = select_scales(
        jnp.zeros((1, 1), dtype=jnp.float64),
        factor,
        metric,
        jnp.array([3.1], dtype=jnp.float64),
        grid,
        half_width=4,
    )
    assert float(off[1]["projected_old"][0]) == 3.0
    assert float(off[0][0]) == 3.0
    # Projection midpoint tie takes the smaller scale.
    midpoint, _ = project_to_grid(grid, jnp.array([3.125], dtype=jnp.float64))
    assert float(midpoint[0]) == 3.0


def test_commit_and_driver_walk_groups():
    key = jax.random.PRNGKey(7)
    k_w, k_x = jax.random.split(key)
    rows, cols, width = 4, 32, 16
    W = jax.random.normal(k_w, (rows, cols), dtype=jnp.float64)
    X = jax.random.normal(k_x, (cols, 20), dtype=jnp.float64)
    H = X @ X.T
    quantized, scales = gptq_quantize(W, H, group_size=width)
    again_w, again_s = gptq_quantize(W, H, group_size=width)
    assert bool(jnp.all(scales == again_s))
    assert bool(jnp.all(quantized == again_w))
    assert quantized.shape == W.shape
    assert scales.shape == (rows, cols // width)
    amax = jnp.max(jnp.abs(W))
    grid = exportable_scales(amax)
    distance = jnp.min(jnp.abs(scales[:, :, None] - grid[None, None, :]), axis=-1)
    assert float(jnp.max(distance)) == 0.0
    levels = jnp.asarray(E2M1_LEVELS, dtype=jnp.float64)
    for group in range(cols // width):
        block = quantized[:, group * width : (group + 1) * width]
        normalized = jnp.abs(block / scales[:, group : group + 1])
        gap = jnp.min(jnp.abs(normalized[..., None] - levels), axis=-1)
        assert float(jnp.max(gap)) <= 1e-8

    # Explicit left-to-right loop, so a missing future update cannot hide.
    factor = damped_inverse_factor(H)
    work = W
    for group in range(cols // width):
        start = group * width
        block = work[:, start : start + width]
        block_factor = factor[start : start + width, start : start + width]
        metric = schur_from_factor(block_factor)
        incoming = initial_group_scales(block, grid)
        winner, _info = select_scales(block, block_factor, metric, incoming, grid)
        assert bool(jnp.all(winner == scales[:, group]))
        work = commit_group(work, factor, start, winner, width)
    assert bool(jnp.all(work == quantized))
