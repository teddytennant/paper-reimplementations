from moara.rank import allocate_ranks, default_block_size, default_r_base


def test_allocate_ranks_budget_and_roles():
    types = ["attn.q", "attn.k", "attn.v", "attn.o", "mlp.gate", "mlp.up", "mlp.down"]
    r_base = 256
    ranks = allocate_ranks(types, r_base)
    assert sum(ranks.values()) == r_base * len(types)
    assert ranks["attn.q"] == r_base // 2
    assert ranks["attn.k"] == r_base // 2
    assert ranks["mlp.down"] == 2 * r_base
    for name in ("attn.v", "attn.o", "mlp.gate", "mlp.up"):
        assert ranks[name] == r_base
    assert all(isinstance(v, int) for v in ranks.values())


def test_allocate_ranks_short_names():
    types = ["q", "k", "v", "o", "gate", "up", "down"]
    r_base = 512
    ranks = allocate_ranks(types, r_base)
    assert sum(ranks.values()) == r_base * len(types)
    assert ranks["q"] == r_base // 2
    assert ranks["k"] == r_base // 2
    assert ranks["down"] == 2 * r_base
    assert ranks["v"] == r_base


def test_allocate_ranks_odd_base_integer_sum():
    types = ["q", "k", "v", "o", "gate", "up", "down"]
    r_base = 255
    ranks = allocate_ranks(types, r_base)
    assert sum(ranks.values()) == r_base * len(types)
    assert all(isinstance(v, int) for v in ranks.values())


def test_default_r_base_and_block_size():
    assert default_r_base(3.5e8) == 256
    assert default_r_base("1B") == 512
    assert default_r_base("7B") == 1024
    assert default_block_size(64, scale="350M") == 32
    assert default_block_size(64, scale="1B") == 64
    assert default_block_size(128, scale="7B") == 128
    assert default_block_size(64) == 64
    assert default_block_size(128) == 128
