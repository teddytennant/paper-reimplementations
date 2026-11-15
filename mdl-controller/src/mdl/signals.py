"""Extract (M, R) from embeddings plus an optional lexical conflict stub."""

from __future__ import annotations

import re
from typing import Optional, Sequence

import numpy as np

from mdl.constants import RELEVANCE_COS_THRESHOLD

_TOKEN = re.compile(r"[a-z]+")
_YES = frozenset({"yes"})
_NEG = frozenset({"no", "not", "never"})
_ANTONYM_PAIRS = (
    ("true", "false"),
    ("increase", "decrease"),
    ("safe", "dangerous"),
    ("correct", "incorrect"),
    ("always", "never"),
    ("positive", "negative"),
    ("accept", "reject"),
    ("success", "failure"),
    ("legal", "illegal"),
    ("healthy", "unhealthy"),
    ("up", "down"),
    ("high", "low"),
    ("good", "bad"),
    ("right", "wrong"),
    ("yes", "no"),
)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def pair_conflict(a: str, b: str) -> bool:
    """Flag a pair on yes/no polarity clash or a small antonym list."""
    ta, tb = _tokens(a), _tokens(b)
    if (ta & _YES and tb & _NEG) or (tb & _YES and ta & _NEG):
        return True
    for x, y in _ANTONYM_PAIRS:
        if (x in ta and y in tb) or (y in ta and x in tb):
            return True
    return False


def conflict_rate(texts: Sequence[str]) -> float:
    """φ = fraction of pairs flagged as conflicting. 0 when fewer than 2 texts."""
    n = len(texts)
    if n < 2:
        return 0.0
    n_pairs = 0
    n_conf = 0
    for i in range(n):
        for j in range(i + 1, n):
            n_pairs += 1
            if pair_conflict(texts[i], texts[j]):
                n_conf += 1
    return n_conf / n_pairs if n_pairs else 0.0


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x if n < eps else x / n
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def extract_signals(
    eq,
    ems,
    texts: Optional[Sequence[Optional[str]]] = None,
    cos_threshold: float = RELEVANCE_COS_THRESHOLD,
):
    """M = max_j cos(e_q, e_m_j); R from relevant-set pairwise cosine and φ.

    If |relevant| < 2, R = 0.5. Else R = clip(mean_pairwise_cos * (1-φ)^2, 0, 1).
    φ = 0 when texts is None.
    """
    eq = _l2_normalize(np.asarray(eq, dtype=np.float64).reshape(-1))
    ems = np.asarray(ems, dtype=np.float64)
    extra = {"n_relevant": 0, "phi": 0.0, "pairwise_mean": None}

    if ems.size == 0:
        return 0.0, 0.5, extra

    ems = np.atleast_2d(ems)
    ems = _l2_normalize(ems)
    cos = ems @ eq
    M = float(np.max(cos)) if cos.size else 0.0

    relevant = np.where(cos > cos_threshold)[0]
    extra["n_relevant"] = int(relevant.size)

    if relevant.size < 2:
        extra["phi"] = 0.0
        return M, 0.5, extra

    rel = ems[relevant]
    gram = rel @ rel.T
    n_rel = rel.shape[0]
    iu = np.triu_indices(n_rel, k=1)
    pair_cos = gram[iu]
    s_bar = float(np.mean(pair_cos)) if pair_cos.size else 0.0
    extra["pairwise_mean"] = s_bar

    phi = 0.0
    if texts is not None:
        rel_texts = [texts[int(j)] for j in relevant]
        if all(t is not None for t in rel_texts):
            phi = conflict_rate(rel_texts)
    extra["phi"] = float(phi)

    R = s_bar * (1.0 - phi) ** 2
    R = float(np.clip(R, 0.0, 1.0))
    return M, R, extra
