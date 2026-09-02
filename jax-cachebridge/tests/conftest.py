import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from cachebridge import layer_select, support, toy  # noqa: E402

SRC_CFG = toy.GQAConfig(n_layers=4, n_kv_heads=4, q_per_kv=2, head_dim=16)
TGT_CFG = toy.GQAConfig(n_layers=4, n_kv_heads=4, q_per_kv=3, head_dim=16)
SEQ_LEN = 48
STRIDE = 4
K_LAYERS = 2


@pytest.fixture(scope="session")
def toy_pair():
    keys = jax.random.split(jax.random.PRNGKey(3), 5)
    src_params = toy.init_source(keys[0], SRC_CFG)
    tgt_params = toy.init_target(keys[1], TGT_CFG)
    plant = toy.init_plant(keys[2], SRC_CFG, TGT_CFG)
    train = toy.make_traces(keys[3], SRC_CFG, TGT_CFG, plant, 16, SEQ_LEN,
                            src_params, tgt_params)
    held = toy.make_traces(keys[4], SRC_CFG, TGT_CFG, plant, 6, SEQ_LEN,
                           src_params, tgt_params)
    sample_pos = jnp.arange(0, SEQ_LEN, STRIDE)
    boundaries = jnp.unique(jnp.int32(jnp.round(jnp.geomspace(8, SEQ_LEN - 1, 6))))
    flat_train = toy.flatten_positions(train, sample_pos)
    flat_held = toy.flatten_positions(held, sample_pos)
    assign = support.identity_assignment(TGT_CFG.n_kv_heads)
    layers, scores = layer_select.select_layers(*flat_train, assign, K_LAYERS)
    return dict(plant=plant, train=train, held=held, flat_train=flat_train,
                flat_held=flat_held, sample_pos=sample_pos, boundaries=boundaries,
                layers=layers, probe_scores=scores, assign=assign,
                src_cfg=SRC_CFG, tgt_cfg=TGT_CFG, k=K_LAYERS)
