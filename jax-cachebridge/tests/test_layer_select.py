import jax.numpy as jnp

from cachebridge import layer_select, support


def test_selector_recovers_the_planted_source_support(toy_pair):
    """Section 3.1's probe ranking should find the layers the toy pair was built from."""
    layers, scores = toy_pair["layers"], toy_pair["probe_scores"]
    planted = toy_pair["plant"].support
    for l, want in enumerate(planted):
        assert tuple(int(j) for j in layers[l]) == want
    assert scores.shape == (toy_pair["tgt_cfg"].n_layers, toy_pair["src_cfg"].n_layers)


def test_selected_layers_score_above_the_rejected_ones(toy_pair):
    layers, scores = toy_pair["layers"], toy_pair["probe_scores"]
    ls = scores.shape[1]
    for l in range(scores.shape[0]):
        chosen = set(int(j) for j in layers[l])
        best_rejected = max(float(scores[l, j]) for j in range(ls) if j not in chosen)
        worst_chosen = min(float(scores[l, j]) for j in chosen)
        assert worst_chosen > best_rejected


def test_shifted_assignment_lowers_the_probe_score_on_the_planted_pairs(toy_pair):
    """The probes are fitted under a(h), so a wrong a(h) must score worse where it counts."""
    sk, sv, tk, tv = toy_pair["flat_train"]
    ht = toy_pair["tgt_cfg"].n_kv_heads
    planted = toy_pair["plant"].support
    aligned = layer_select.probe_scores(sk, sv, tk, tv, support.identity_assignment(ht))
    for shift in range(1, ht):
        shifted = layer_select.probe_scores(
            sk, sv, tk, tv, support.cyclic_assignment(ht, shift))
        for l, sources in enumerate(planted):
            for j in sources:
                assert float(aligned[l, j]) > float(shifted[l, j]) + 0.2
        assert float(jnp.mean(aligned)) > float(jnp.mean(shifted)) + 0.1
