"""KL distillation on converted byte distributions (Appendix A.3, byte-level)."""

from __future__ import annotations

import jax.numpy as jnp
from jax.nn import log_softmax


def kl_byte_distill(teacher_byte_probs, student_byte_logits):
    """Sum over gold byte positions of KL(teacher || student).

    teacher_byte_probs: (T, n_bytes) teacher byte distributions after F.1.
    student_byte_logits: (T, n_bytes) student byte logits (softmaxed here).
    """
    teacher = jnp.asarray(teacher_byte_probs)
    log_q = log_softmax(jnp.asarray(student_byte_logits), axis=-1)
    log_p = jnp.where(teacher > 0, jnp.log(teacher), 0.0)
    term = teacher * (log_p - log_q)
    term = jnp.where(teacher > 0, term, 0.0)
    return jnp.sum(term)
