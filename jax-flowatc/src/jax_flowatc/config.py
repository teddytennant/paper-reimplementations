"""Model widths. Tiny defaults for CPU tests; paper_tiny is flow_tiny in Table 2."""

from __future__ import annotations

from typing import NamedTuple


class FlowATCConfig(NamedTuple):
    n_layers: int = 2
    n_heads: int = 2
    d_model: int = 32
    t_obs: int = 4
    t_fut: int = 4
    state_dim: int = 6
    ffn_mult: int = 4
    dropout: float = 0.0
    ln_eps: float = 1e-6
    time_max_period: float = 10000.0

    @property
    def seq_len(self) -> int:
        return self.t_obs + self.t_fut

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


def tiny_config(**overrides) -> FlowATCConfig:
    """CPU test default: 2 layers, 2 heads, d=32, 4+4 tokens."""
    return FlowATCConfig(**overrides)


def paper_tiny_config(**overrides) -> FlowATCConfig:
    """flow_tiny in Table 2: 5 layers, 4 heads, d=128. Not used in tests."""
    cfg = FlowATCConfig(
        n_layers=5,
        n_heads=4,
        d_model=128,
        t_obs=43,
        t_fut=43,
        state_dim=6,
        ffn_mult=4,
        dropout=0.1,
    )
    if overrides:
        cfg = cfg._replace(**overrides)
    return cfg
