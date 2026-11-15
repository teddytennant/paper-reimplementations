"""Public decide() / decide_from_memories() API (Algorithm 1)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from mdl.actions import ADOPT, action_from_cfinal
from mdl.constants import BIAS, NMAX, NMIN, QR_SEED, THETA
from mdl.fusion import confidence_consistency, fuse
from mdl.signals import extract_signals
from mdl.subspaces import build_subspaces


@dataclass(frozen=True)
class Decision:
    action: str
    C: float
    alpha: float
    Cfinal: float
    M: float
    R: float
    A: float
    gA: float
    va_norm: float
    adopt: bool


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def decide(
    M: float,
    R: float,
    A: float,
    *,
    nmin: float = NMIN,
    nmax: float = NMAX,
    theta=THETA,
    qr_seed: int = QR_SEED,
    bias: float = BIAS,
) -> Decision:
    """Geometric memory decision from scalar signals (M, R, A) in [0, 1]."""
    M = _clip01(float(M))
    R = _clip01(float(R))
    A = _clip01(float(A))
    sub = build_subspaces(seed=int(qr_seed))
    parts = fuse(M, R, A, sub, bias=bias)
    C, alpha, Cfinal = confidence_consistency(
        parts["vmeta"], parts["vwm"], nmin=nmin, nmax=nmax
    )
    C = float(C)
    alpha = float(alpha)
    Cfinal = float(Cfinal)
    action = action_from_cfinal(Cfinal, theta=theta)
    return Decision(
        action=action,
        C=C,
        alpha=alpha,
        Cfinal=Cfinal,
        M=M,
        R=R,
        A=A,
        gA=float(parts["gA"]),
        va_norm=float(parts["va_norm"]),
        adopt=action in ADOPT,
    )


def decide_from_memories(
    eq,
    ems,
    A: float,
    texts: Optional[Sequence[Optional[str]]] = None,
    *,
    nmin: float = NMIN,
    nmax: float = NMAX,
    theta=THETA,
    qr_seed: int = QR_SEED,
    bias: float = BIAS,
) -> Decision:
    """Extract (M, R) from embeddings (and optional texts), then decide."""
    M, R, _ = extract_signals(eq, ems, texts=texts)
    return decide(
        M, R, A, nmin=nmin, nmax=nmax, theta=theta, qr_seed=qr_seed, bias=bias
    )
