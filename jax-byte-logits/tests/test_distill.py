"""KL(byte_teacher || byte_student) on converted distributions."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jax_byte_logits import byte_probabilities, kl_byte_distill, token_byte_tables

TOKENS = ["isu", "isk", "is", "impl", "imap", "ino", "mit", "max", "new"]
PROBS = np.asarray(
    [0.4, 0.1, 0.1, 0.05, 0.05, 0.1, 0.05, 0.05, 0.1], dtype=np.float64
)


def _teacher():
    table, lengths = token_byte_tables(TOKENS, use_eot=False)
    return byte_probabilities(PROBS, table, lengths, "isu", use_eot=False)


def test_kl_copy_near_zero():
    teacher = _teacher()
    student_logits = jnp.log(jnp.maximum(teacher, 1e-12))
    kl = float(kl_byte_distill(teacher, student_logits))
    assert kl < 1e-5


def test_kl_uniform_larger_than_copy():
    teacher = _teacher()
    copy_logits = jnp.log(jnp.maximum(teacher, 1e-12))
    uniform_logits = jnp.zeros_like(teacher)
    kl_copy = float(kl_byte_distill(teacher, copy_logits))
    kl_uniform = float(kl_byte_distill(teacher, uniform_logits))
    assert kl_uniform > kl_copy
    assert kl_uniform > 1.0


def test_kl_sums_over_positions():
    teacher = _teacher()
    uniform_logits = jnp.zeros_like(teacher)
    full = float(kl_byte_distill(teacher, uniform_logits))
    first = float(kl_byte_distill(teacher[:1], uniform_logits[:1]))
    rest = float(kl_byte_distill(teacher[1:], uniform_logits[1:]))
    np.testing.assert_allclose(full, first + rest, rtol=1e-5, atol=1e-6)
