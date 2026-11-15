import numpy as np

from refinemix.schedule import alpha, alpha_ideal
from refinemix.theory import (
    ToyDomain,
    enumerate_terminal,
    overlap_C,
    overlap_grid,
    source_probability,
    theorem1_rhs,
    time_grid,
)


def _disjoint():
    return ToyDomain(target=[(0, 0)], source=[(1, 1)], mask_id=2)


def _overlap(length):
    if length == 2:
        return ToyDomain(
            target=[(0, 0), (0, 1)],
            source=[(1, 0), (1, 1)],
            mask_id=2,
        )
    if length == 3:
        return ToyDomain(
            target=[(0, 0, 0), (0, 0, 1), (0, 1, 0)],
            source=[(1, 1, 1), (1, 1, 0), (1, 0, 1)],
            mask_id=2,
        )
    raise ValueError(length)


def test_theorem1_equality_on_enumerable_domain():
    # Idealized kernel: alpha(1) = 0, not the eps-truncated experimental schedule.
    assert float(alpha_ideal(1.0)) == 0.0
    assert abs(float(alpha(1.0)) - 1e-3) < 1e-6

    settings = [
        dict(lam=0.3, tmax=0.4, domain=_disjoint(), analytical=True),
        dict(lam=0.5, tmax=0.6, domain=_overlap(2), analytical=False),
        dict(lam=0.25, tmax=0.3, domain=_overlap(3), analytical=False),
    ]
    saw_partial_overlap = False
    for setting in settings:
        lam = setting["lam"]
        tmax = setting["tmax"]
        domain = setting["domain"]
        times = time_grid(tmax, n=9 if domain.length == 2 else 7)
        mu = enumerate_terminal(domain, lam, tmax, times)
        total = sum(mu.values())
        assert abs(total - 1.0) < 1e-8
        prob = source_probability(mu, domain)
        c_tmax = overlap_C(domain, tmax, lam)
        assert 0.0 < c_tmax < 1.0
        saw_partial_overlap = True
        expected = theorem1_rhs(lam, c_tmax)
        assert abs(prob - expected) < 1e-7, (lam, tmax, prob, expected, c_tmax)
        if setting["analytical"]:
            # Token-disjoint single sequences: C(t) = t^L * (1 - lam), so P = lam * t^L.
            analytical_c = (tmax ** domain.length) * (1.0 - lam)
            assert abs(c_tmax - analytical_c) < 1e-9
            assert abs(prob - lam * (tmax ** domain.length)) < 1e-8
        for t_prime in np.linspace(tmax, 1.0, 6):
            bound = theorem1_rhs(lam, overlap_C(domain, float(t_prime), lam))
            assert prob <= bound + 1e-8
    assert saw_partial_overlap


def test_lemma5_overlap_is_nondecreasing():
    for domain, lam in ((_disjoint(), 0.3), (_overlap(2), 0.4), (_overlap(3), 0.55)):
        times = np.linspace(0.0, 1.0, 21)
        values = overlap_grid(domain, lam, times)
        for earlier, later in zip(values, values[1:]):
            assert later >= earlier - 1e-10
        assert values[0] == 0.0 or abs(values[0]) < 1e-12
        assert values[-1] > values[0]
