"""Four-level action mapping (Sec. 3.4 / Fig. 4)."""

from __future__ import annotations

from mdl.constants import THETA

ACTIONS = ("Active", "Supp", "Silent", "Opt-Out")
ADOPT = frozenset({"Active", "Supp"})


def action_from_cfinal(Cfinal: float, theta=THETA) -> str:
    """Map C_final to {Active, Supp, Silent, Opt-Out} with θ = (θ0, θ1, θ2)."""
    t0, t1, t2 = tuple(theta)
    c = float(Cfinal)
    if c >= t0:
        return "Active"
    if c >= t1:
        return "Supp"
    if c >= t2:
        return "Silent"
    return "Opt-Out"
