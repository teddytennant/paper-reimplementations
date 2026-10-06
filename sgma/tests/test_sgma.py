"""CPU tests for the structure-guided tokenizer, damped canvas, and tiny MAE."""

import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sgma.damped import canvas, damped_variance, increment
from sgma.mae import init_params, mae_loss, prepare, train_step
from sgma.masking import structure_mask, uniform_mask
from sgma.tokenizer import split_probs, splittable_probs, tokenize, tokenize_with_depths


def test_softmax_probs_sum_to_one():
    values = np.array([0.2, 3.0, 1.5, 0.0, 8.0])
    probs = split_probs(values, tau=1.0)
    assert probs.shape == values.shape
    assert abs(float(probs.sum()) - 1.0) < 1e-8
    assert np.all(probs > 0)
    assert probs[np.argmax(values)] == np.max(probs)

    salience = np.zeros((8, 8))
    salience[:4, :4] = 5.0
    tokens, _depths = tokenize_with_depths(salience, 4, deterministic=True)
    live, _eligible = splittable_probs(salience, tokens, tau=0.7)
    assert abs(float(live.sum()) - 1.0) < 1e-8


def test_deterministic_tokenize_reproducible_and_grows_by_three():
    salience = np.random.default_rng(0).random((16, 16))
    first = tokenize(salience, 13, deterministic=True, seed=0)
    second = tokenize(salience, 13, deterministic=True, seed=99)
    assert first == second
    assert len(first) == 13
    assert all(len(token) == 4 for token in first)

    previous = tokenize(salience, 1, deterministic=True)
    assert len(previous) == 1
    for step in range(1, 5):
        target = 1 + 3 * step
        tokens = tokenize(salience, target, deterministic=True)
        assert len(tokens) == target
        assert len(tokens) - len(tokenize(salience, target - 3, deterministic=True)) == 3

    assert len(tokenize(salience, 2, deterministic=True)) == 1
    assert len(tokenize(salience, 5, deterministic=True)) == 4
    assert len(tokenize(salience, 12, deterministic=True)) == 10

    tiny = np.ones((2, 2))
    assert len(tokenize(tiny, 100, deterministic=True)) == 4


def test_split_prefers_high_salience_and_reports_depths():
    salience = np.zeros((8, 8))
    salience[:4, :4] = 10.0
    tokens, depths = tokenize_with_depths(salience, 7, deterministic=True)
    assert len(tokens) == 7
    assert depths.count(1) == 3
    assert depths.count(2) == 4
    assert 0 not in depths
    deep = [token for token, depth in zip(tokens, depths) if depth == 2]
    assert len(deep) == 4
    for r0, r1, c0, c1 in deep:
        assert r0 >= 0 and c0 >= 0 and r1 <= 4 and c1 <= 4


def test_damped_variance_all_ones():
    structure = np.ones((8, 8))
    assert abs(damped_variance(structure, (0, 8, 0, 8), 0) - 1.0) < 1e-6
    depth1 = 1.0 + 1.0 / np.sqrt(2.0)
    depth2 = depth1 + 1.0 / np.sqrt(3.0)
    assert abs(damped_variance(structure, (0, 4, 0, 4), 1) - depth1) < 1e-6
    assert abs(damped_variance(structure, (0, 2, 0, 2), 2) - depth2) < 1e-6


def test_tree_depths_and_nonconstant_map():
    structure = np.ones((8, 8))
    tokens, depths = tokenize_with_depths(structure, 7, deterministic=True)
    assert depths.count(2) == 4 and depths.count(1) == 3
    variances = canvas(structure, tokens, depths)
    assert variances.shape == (7,)
    for value, depth in zip(variances, depths):
        expected = sum(1.0 / np.sqrt(j + 1) for j in range(depth + 1))
        assert abs(float(value) - expected) < 1e-6

    depth1_token = next(token for token, depth in zip(tokens, depths) if depth == 1)
    depth2_token = next(token for token, depth in zip(tokens, depths) if depth == 2)
    variance_d1 = damped_variance(structure, depth1_token, 1)
    variance_d2 = damped_variance(structure, depth2_token, 2)
    parent_mean = 1.0
    assert abs((variance_d2 - variance_d1) - (1.0 / np.sqrt(3.0)) * parent_mean) < 1e-6

    mixed = np.zeros((8, 8))
    mixed[:4, :4] = 4.0
    assert abs(float(mixed.mean()) - 1.0) < 1e-6
    mixed_var = damped_variance(mixed, (0, 4, 0, 4), 1)
    assert abs(mixed_var - (1.0 / 1.0 + 4.0 / np.sqrt(2.0))) < 1e-6


def test_sublinear_increments():
    assert increment(0) > increment(5)
    structure = np.ones((64, 64))
    inc_0 = damped_variance(structure, (0, 32, 0, 32), 1) - damped_variance(
        structure, (0, 64, 0, 64), 0
    )
    inc_5 = damped_variance(structure, (0, 1, 0, 1), 6) - damped_variance(
        structure, (0, 2, 0, 2), 5
    )
    assert inc_0 > inc_5
    assert abs(inc_0 - increment(1)) < 1e-6
    assert abs(inc_5 - increment(6)) < 1e-6


def test_uniform_mask_keeps_both_roles():
    mask = uniform_mask(13, ratio=0.75, seed=0)
    assert mask.dtype == np.bool_
    assert mask.shape == (13,)
    assert mask.any() and (~mask).any()
    assert abs(float(mask.mean()) - 0.75) < 0.15
    again = uniform_mask(13, ratio=0.75, seed=0)
    assert np.array_equal(mask, again)


def test_structure_mask_prefers_high_variance():
    variances = np.array([10.0, 0.0, 0.0, 0.0])
    hits = 0
    for seed in range(40):
        mask = structure_mask(variances, ratio=0.25, seed=seed)
        assert int(mask.sum()) == 1
        if bool(mask[0]):
            hits += 1
    assert hits >= 35

    zeros = structure_mask(np.zeros(8), ratio=0.75, seed=3)
    uniform = uniform_mask(8, ratio=0.75, seed=3)
    assert np.array_equal(zeros, uniform)


def test_mae_loss_ignores_visible_targets_and_uses_masked_targets():
    salience = np.ones((16, 16))
    salience[:8, :8] = 4.0
    images = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16, 1))
    params = init_params(jax.random.PRNGKey(0), channels=1)
    key = jax.random.PRNGKey(2)

    loss0 = mae_loss(params, images, salience, key)
    tokens, _depths, mask, _variances = prepare(salience, key, target_n=13, ratio=0.75)
    visible = int(np.flatnonzero(~mask)[0])
    masked = int(np.flatnonzero(mask)[0])

    def perturb(index):
        targets = np.array(images)
        r0, _r1, c0, _c1 = tokens[index]
        targets[:, r0, c0, :] += 5.0
        return jnp.asarray(targets)

    loss_visible = mae_loss(params, images, salience, key, targets=perturb(visible))
    loss_masked = mae_loss(params, images, salience, key, targets=perturb(masked))
    assert abs(float(loss_visible) - float(loss0)) < 1e-6
    assert abs(float(loss_masked) - float(loss0)) > 1e-4


def test_train_step_finite_loss_and_nonzero_encoder_grad():
    salience = np.ones((16, 16), dtype=np.float32)
    salience[:8, :8] = 3.0
    images = jax.random.normal(jax.random.PRNGKey(4), (2, 16, 16, 1))
    params = init_params(0, channels=1)
    key = jax.random.PRNGKey(5)
    tokens, _depths, mask, _variances = prepare(salience, key)

    from sgma.mae import _forward_loss

    def loss_fn(current):
        return _forward_loss(current, images, images, tokens, mask)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    leaves = jax.tree_util.tree_leaves(grads["encoder"])
    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))
    assert bool(jnp.isfinite(loss))
    assert bool(jnp.isfinite(grad_norm))
    assert float(grad_norm) > 0.0

    new_params, step_loss = train_step(params, images, salience, key)
    assert bool(jnp.isfinite(step_loss))
    moved = False
    for before, after in zip(
        jax.tree_util.tree_leaves(params["encoder"]),
        jax.tree_util.tree_leaves(new_params["encoder"]),
    ):
        if not np.allclose(np.asarray(before), np.asarray(after)):
            moved = True
            break
    assert moved


def test_no_network_imports():
    root = Path(__file__).resolve().parents[1] / "sgma"
    banned = {"requests", "urllib", "httpx", "socket", "wget"}
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module.split(".")[0]] if node.module else []
            else:
                continue
            assert banned.isdisjoint(names)
