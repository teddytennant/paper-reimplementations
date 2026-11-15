"""Quadtree tokenizer over a 2D salience map (section 3.1).

A token is a half-open rectangle ``(r0, r1, c0, c1)``. The root is the full
map. Each split replaces one eligible token with its four quadrants, so the
token count increases by 3. Counts not of the form ``1 + 3k`` are not
reachable; tokenization stops at the largest reachable count that does not
exceed ``target_n``.
"""

from __future__ import annotations

import numpy as np


def token_value(salience, token):
    """V(token): sum of salience over the half-open rectangle."""
    r0, r1, c0, c1 = token
    return float(np.sum(np.asarray(salience)[r0:r1, c0:c1]))


def can_split(token):
    r0, r1, c0, c1 = token
    return (r1 - r0) >= 2 and (c1 - c0) >= 2


def quadrants(token):
    r0, r1, c0, c1 = (int(x) for x in token)
    rm = r0 + (r1 - r0) // 2
    cm = c0 + (c1 - c0) // 2
    return (
        (r0, rm, c0, cm),
        (r0, rm, cm, c1),
        (rm, r1, c0, cm),
        (rm, r1, cm, c1),
    )


def split_probs(values, tau=1.0):
    """pi = softmax(V / tau) over eligible tokens. Rows sum to 1."""
    logits = np.asarray(values, dtype=np.float64) / float(tau)
    logits = logits - np.max(logits)
    weights = np.exp(logits)
    return weights / np.sum(weights)


def splittable_probs(salience, tokens, tau=1.0):
    """Softmax over tokens that can still split. Returns (probs, indices)."""
    eligible = [i for i, token in enumerate(tokens) if can_split(token)]
    if not eligible:
        return np.zeros(0, dtype=np.float64), eligible
    values = [token_value(salience, tokens[i]) for i in eligible]
    return split_probs(values, tau), eligible


def tokenize_with_depths(salience, target_n, tau=1.0, deterministic=True, seed=0):
    """Return ``(tokens, depths)`` for a quadtree stopped at ``target_n``.

    Root depth is 0. Each split increments depth. Deterministic mode always
    splits the eligible token with the largest V (tau -> 0); ties keep the
    earliest token.
    """
    salience = np.asarray(salience, dtype=np.float64)
    if salience.ndim != 2:
        raise ValueError(f"salience must be (H, W), got {salience.shape}")
    height, width = salience.shape
    tokens = [(0, int(height), 0, int(width))]
    depths = [0]
    if target_n < 1:
        return [], []

    rng = np.random.default_rng(int(seed))
    while len(tokens) + 3 <= int(target_n):
        eligible = [i for i, token in enumerate(tokens) if can_split(token)]
        if not eligible:
            break
        values = np.asarray(
            [token_value(salience, tokens[i]) for i in eligible], dtype=np.float64
        )
        if deterministic:
            choice = eligible[int(np.argmax(values))]
        else:
            probs = split_probs(values, tau)
            choice = eligible[int(rng.choice(len(eligible), p=probs))]
        parent = tokens.pop(choice)
        parent_depth = depths.pop(choice)
        children = quadrants(parent)
        for offset, child in enumerate(children):
            tokens.insert(choice + offset, child)
            depths.insert(choice + offset, parent_depth + 1)
    return tokens, depths


def tokenize(salience, target_n, tau=1.0, deterministic=True, seed=0):
    """Tokenize ``salience`` into at most ``target_n`` quadtree rectangles."""
    tokens, _depths = tokenize_with_depths(
        salience,
        target_n,
        tau=tau,
        deterministic=deterministic,
        seed=seed,
    )
    return tokens
