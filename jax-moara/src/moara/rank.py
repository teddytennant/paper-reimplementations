"""Static module-aware rank allocation (Algorithm 1 Phase 3, Appendix D.2)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

R_BASE_350M = 256
R_BASE_1B = 512
R_BASE_7B = 1024

DEFAULT_DONORS = ("attn.q", "attn.k")
DEFAULT_RECEIVERS = ("mlp.down",)

_CANON_ALIASES = {
    "q": "attn.q",
    "query": "attn.q",
    "q_proj": "attn.q",
    "wq": "attn.q",
    "attn.q": "attn.q",
    "attn_q": "attn.q",
    "k": "attn.k",
    "key": "attn.k",
    "k_proj": "attn.k",
    "wk": "attn.k",
    "attn.k": "attn.k",
    "attn_k": "attn.k",
    "v": "attn.v",
    "value": "attn.v",
    "v_proj": "attn.v",
    "wv": "attn.v",
    "attn.v": "attn.v",
    "attn_v": "attn.v",
    "o": "attn.o",
    "out": "attn.o",
    "output": "attn.o",
    "o_proj": "attn.o",
    "wo": "attn.o",
    "attn.o": "attn.o",
    "attn_o": "attn.o",
    "gate": "mlp.gate",
    "gate_proj": "mlp.gate",
    "w1": "mlp.gate",
    "mlp.gate": "mlp.gate",
    "mlp_gate": "mlp.gate",
    "up": "mlp.up",
    "up_proj": "mlp.up",
    "w3": "mlp.up",
    "mlp.up": "mlp.up",
    "mlp_up": "mlp.up",
    "down": "mlp.down",
    "down_proj": "mlp.down",
    "w2": "mlp.down",
    "mlp.down": "mlp.down",
    "mlp_down": "mlp.down",
}


def _canonical(name: str) -> str:
    raw = str(name).strip().lower().replace("-", "_")
    if raw in _CANON_ALIASES:
        return _CANON_ALIASES[raw]
    tail = raw.split(".")[-1]
    if tail in _CANON_ALIASES:
        return _CANON_ALIASES[tail]
    for prefix in ("attn_", "mlp_", "ffn_"):
        if tail.startswith(prefix):
            clipped = tail[len(prefix) :]
            if clipped in _CANON_ALIASES:
                return _CANON_ALIASES[clipped]
    return raw


def _parse_n_params(scale) -> float:
    if isinstance(scale, (int, float)):
        return float(scale)
    s = str(scale).strip().lower().replace("_", "")
    if s.endswith("m"):
        return float(s[:-1]) * 1e6
    if s.endswith("b"):
        return float(s[:-1]) * 1e9
    return float(s)


def default_r_base(n_params_hint) -> int:
    """GaLore-convention rank: 256 at 350M, 512 at 1B, 1024 at 7B."""
    n = _parse_n_params(n_params_hint)
    if n <= 5e8:
        return R_BASE_350M
    if n <= 2e9:
        return R_BASE_1B
    return R_BASE_7B


def default_block_size(d_head: int, scale=None) -> int:
    """Appendix D.2: B=32 (350M, d_head=64), B=64 (1B, d_head=64), B=128 (7B, d_head=128).

    Without ``scale``, return ``d_head`` (neighborhood of d_head).
    """
    if scale is not None:
        n = _parse_n_params(scale)
        if n <= 5e8:
            return 32
        if n <= 2e9:
            return 64
        return 128
    return int(d_head)


def allocate_ranks(
    module_types: Sequence[str],
    r_base: int,
    donors: Sequence[str] = DEFAULT_DONORS,
    receivers: Sequence[str] = DEFAULT_RECEIVERS,
    delta_r: float | None = None,
) -> dict[str, int]:
    """Static donor/receiver rank allocation.

    D = {attn.q, attn.k}, R = {mlp.down}, Δr = r_base / 2.

        r_τ ← r_base − Δr              for τ ∈ D
        r_τ ← r_base + Δr · |D|/|R|    for τ ∈ R
        r_τ ← r_base                   otherwise

    Integer ranks with largest-remainder rounding so sum_τ r_τ = |types| * r_base.
    """
    types = list(module_types)
    if not types:
        return {}
    r_base = int(r_base)
    if delta_r is None:
        delta_r = r_base / 2.0
    donor_set = {_canonical(d) for d in donors}
    recv_set = {_canonical(r) for r in receivers}
    canon = [_canonical(t) for t in types]
    n_d = sum(1 for c in canon if c in donor_set)
    n_r = sum(1 for c in canon if c in recv_set)
    extra = (delta_r * n_d / n_r) if n_r else 0.0

    floats: list[float] = []
    for c in canon:
        if c in donor_set:
            floats.append(float(r_base) - float(delta_r))
        elif c in recv_set:
            floats.append(float(r_base) + float(extra))
        else:
            floats.append(float(r_base))

    total = int(len(types) * r_base)
    floors = [int(x // 1) for x in floats]
    remainders = [x - f for x, f in zip(floats, floors)]
    need = total - sum(floors)
    order = sorted(range(len(floors)), key=lambda i: remainders[i], reverse=True)
    ranks = list(floors)
    if need > 0:
        for i in order[:need]:
            ranks[i] += 1
    elif need < 0:
        for i in reversed(order[: (-need)]):
            ranks[i] -= 1
        # If largest-remainder underflowed, peel from the end.
        idx = len(ranks) - 1
        while sum(ranks) > total and idx >= 0:
            if ranks[idx] > 0:
                ranks[idx] -= 1
            idx -= 1
        idx = 0
        while sum(ranks) < total and idx < len(ranks):
            ranks[idx] += 1
            idx += 1

    # Exact budget: if still off (e.g. empty types already handled), fix last.
    drift = total - sum(ranks)
    if drift != 0 and ranks:
        ranks[-1] += drift
    return {name: int(r) for name, r in zip(types, ranks)}


def ranks_from_mapping(
    type_counts: Mapping[str, int],
    r_base: int,
    **kwargs,
) -> dict[str, int]:
    """Allocate once per type, then the caller can broadcast by count."""
    return allocate_ranks(list(type_counts.keys()), r_base, **kwargs)
