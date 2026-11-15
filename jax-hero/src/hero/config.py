"""Tiny HeRo decoder configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeroConfig:
    d_model: int = 32
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 64
    d_r: int = 8
    d_h: int = 16
    d_a: int = 16
    d_m: int = 8
    d_cmp: int = 8
    vocab_size: int = 64
    tau_exec: float = 0.5
    t_hist: float = 1.0
    eps: float = 1e-6
    alpha_t: float = 0.1
    ln_eps: float = 1e-5

    @property
    def d_s(self) -> int:
        return self.d_h + self.d_a

    @property
    def n_routed(self) -> int:
        return self.n_layers

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads
