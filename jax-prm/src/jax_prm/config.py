"""PRMConfig: decoder-only transformer + Persistent Recurrent Memory."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PRMConfig:
    """Hyperparameters for the decoder-only transformer and the PRM module.

    Defaults follow the task's paper-scale kernel (d_model=256, 6 layers),
    not the TinyStories ablation table in the PDF (d_model=192).
    """

    n_layers: int = 6
    d_model: int = 256
    n_heads: int = 8
    d_ff: int = 1024
    vocab_size: int = 8000
    context: int = 256
    n_slots: int = 16
    d_mem: int = 128
    n_mem_heads: int = 4
    dropout: float = 0.1
    aux_lambda: float = 0.0
    use_prm: bool = True
    ln_eps: float = 1e-5

    @property
    def prm_enabled(self) -> bool:
        return bool(self.use_prm and self.n_slots > 0)

    @property
    def split_after(self) -> int:
        """0-based index of the last lower-half layer (PRM runs after this block)."""
        return max(self.n_layers // 2 - 1, 0)

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads

    @property
    def d_mem_head(self) -> int:
        return self.d_model // self.n_mem_heads
