from refinemix.neff import kish_neff


def test_neff_equals_total_at_kish_lambda():
    for n_target, n_source in [(10, 30), (1, 1), (7, 13), (100, 4), (3, 9)]:
        lam = n_source / (n_target + n_source)
        neff = kish_neff(lam, n_target, n_source)
        assert abs(neff - (n_target + n_source)) < 1e-8
