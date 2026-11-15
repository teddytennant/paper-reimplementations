"""Causal dyadic retrieval and the count reader (no learned parameters)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from bridgemem.constants import K, N_BINS, WINDOW


def recency_bin_np(delta_t):
    """NumPy twin of kernel.recency_bin (edges {0, 1, 4, 16, 64})."""
    delta_t = np.asarray(delta_t)
    cuts = np.array([1, 4, 16, 64], dtype=delta_t.dtype)
    return np.clip(np.searchsorted(cuts, delta_t, side="left"), 0, N_BINS - 1)


def _as_arrays(events):
    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("events must be (N, 4) as (s, r, o, t)")
    return (
        events[:, 0].astype(np.int32),
        events[:, 1].astype(np.int32),
        events[:, 2].astype(np.int32),
        events[:, 3].astype(np.int32),
    )


def retrieve_pair(events, actor, candidate, t, k=K, window=WINDOW):
    """Events on ordered pair (actor, candidate) with t_j < t and t_j > t-window.

    Direction d_j = 0 if actor -> candidate, 1 if candidate -> actor.
    Keeps the K most recent events. Never uses time >= t.
    """
    subj, rel, obj, time = _as_arrays(events)
    forward = (subj == actor) & (obj == candidate)
    reverse = (subj == candidate) & (obj == actor)
    causal = (time < t) & (time > t - window)
    sel = np.where((forward | reverse) & causal)[0]
    empty = {
        "relation": np.zeros((k,), dtype=np.int32),
        "direction": np.zeros((k,), dtype=np.int32),
        "delta_t": np.zeros((k,), dtype=np.int32),
        "bins": np.zeros((k,), dtype=np.int32),
        "mask": np.zeros((k,), dtype=bool),
        "time": np.zeros((k,), dtype=np.int32),
    }
    if sel.size == 0:
        return empty
    order = np.argsort(-time[sel], kind="stable")
    sel = sel[order[:k]]
    n = sel.size
    direction = np.where(subj[sel] == actor, 0, 1).astype(np.int32)
    delta = (t - time[sel]).astype(np.int32)
    out = empty
    out["relation"][:n] = rel[sel]
    out["direction"][:n] = direction
    out["delta_t"][:n] = delta
    out["bins"][:n] = recency_bin_np(delta)
    out["mask"][:n] = True
    out["time"][:n] = time[sel]
    return out


def stack_candidates(events, actor, candidates, t, k=K, window=WINDOW):
    """Retrieve B_{a,c,t} for every candidate. Arrays shaped (C, K)."""
    rows = [retrieve_pair(events, actor, int(c), t, k=k, window=window) for c in candidates]
    return {key: np.stack([r[key] for r in rows], axis=0) for key in rows[0]}


def relation_prior(events, n_rel, n_dir=None):
    """Laplace-smoothed marginal π over 2|R| directed relations."""
    if n_dir is None:
        n_dir = 2 * n_rel
    _, rel, _, _ = _as_arrays(events)
    counts = np.bincount(rel.astype(np.int64), minlength=n_dir).astype(np.float64)
    # Training events are stored as original relations (forward ids 0..|R|-1).
    # Reverse ids |R|..2|R|-1 stay on the Laplace floor unless filled.
    pi = (counts + 1.0) / (counts.sum() + n_dir)
    return pi


def fit_count_table(events, n_rel, k=K, window=WINDOW):
    """Smoothed p_ct(r | d, r_j, b) from the causal training prefix.

    For each event used as a next-relation target, every strictly earlier
    dyadic event within the same window/K contributes one count to
    (d_j, r_j, b(Δt), r_target). Laplace add-1 over 2|R| directed classes.
    """
    n_dir = 2 * n_rel
    counts = np.zeros((2, n_rel, N_BINS, n_dir), dtype=np.float64)
    subj, rel, obj, time = _as_arrays(events)
    dyads = defaultdict(list)
    for i in range(subj.shape[0]):
        s, r, o, t = int(subj[i]), int(rel[i]), int(obj[i]), int(time[i])
        if s == o:
            continue
        key = (s, o) if s < o else (o, s)
        dyads[key].append(i)
    ev = np.stack([subj, rel, obj, time], axis=1)
    for idx in dyads.values():
        idx = np.asarray(idx)
        idx = idx[np.argsort(time[idx], kind="stable")]
        for pos, i in enumerate(idx):
            t = int(time[i])
            actor = int(subj[i])
            cand = int(obj[i])
            target = int(rel[i])  # forward directed id
            if target < 0 or target >= n_rel:
                continue
            prefix = ev[idx[:pos]]
            if prefix.size == 0:
                continue
            tok = retrieve_pair(prefix, actor, cand, t, k=k, window=window)
            m = tok["mask"]
            if not np.any(m):
                continue
            d = tok["direction"][m]
            r_j = tok["relation"][m]
            b = tok["bins"][m]
            for dj, rj, bj in zip(d, r_j, b):
                if 0 <= rj < n_rel:
                    counts[int(dj), int(rj), int(bj), target] += 1.0
    token_n = counts.sum(axis=-1)
    p_ct = (counts + 1.0) / (token_n[..., None] + n_dir)
    return p_ct, token_n


def training_examples(events, n_rel, k=K, window=WINDOW):
    """Next-relation examples: prefix tokens and directed target, causal only."""
    subj, rel, obj, time = _as_arrays(events)
    examples = []
    dyads = defaultdict(list)
    for i in range(subj.shape[0]):
        s, o = int(subj[i]), int(obj[i])
        if s == o:
            continue
        key = (s, o) if s < o else (o, s)
        dyads[key].append(i)
    ev = np.stack([subj, rel, obj, time], axis=1)
    for idx in dyads.values():
        idx = np.asarray(idx)
        idx = idx[np.argsort(time[idx], kind="stable")]
        for pos, i in enumerate(idx):
            t = int(time[i])
            actor = int(subj[i])
            cand = int(obj[i])
            target = int(rel[i])
            prefix = ev[idx[:pos]]
            tok = retrieve_pair(prefix, actor, cand, t, k=k, window=window)
            if not np.any(tok["mask"]):
                continue
            examples.append((tok, target))
    return examples
