"""Token-logit to byte-logit conversion (arXiv:2609.12303)."""

from jax_byte_logits.convert import (
    EOT,
    byte_probabilities,
    from_logits,
    token_byte_tables,
)
from jax_byte_logits.distill import kl_byte_distill

__all__ = [
    "EOT",
    "byte_probabilities",
    "from_logits",
    "kl_byte_distill",
    "token_byte_tables",
]
