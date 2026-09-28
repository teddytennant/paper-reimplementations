"""Tiny CPU/GPU smoke: 8 entities, 4 relations. Writes results.json."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from bridgemem.backbone import random_scores
from bridgemem.kernel import (
    attention_pool,
    event_embedding,
    init_params,
    log_scores,
    neural_log_probs,
)
from bridgemem.memory import fit_count_table, relation_prior, stack_candidates, training_examples
from bridgemem.metrics import filtered_mrr, hits_at_k


N_ENT = 8
N_REL = 4


def make_toy(rng: np.random.Generator):
    events = []
    # Planted dyadic cycles on (0,1), (2,3), (4,5), (6,7).
    for a in (0, 2, 4, 6):
        c = a + 1
        for t in range(1, 41):
            r = (t - 1) % N_REL
            events.append((a, r, c, t))
    # Noise on other ordered pairs, strictly mixed times.
    for _ in range(24):
        a, c = rng.integers(0, N_ENT, size=2)
        if a == c or {a, c} in ({0, 1}, {2, 3}, {4, 5}, {6, 7}):
            continue
        r = int(rng.integers(0, N_REL))
        t = int(rng.integers(1, 41))
        events.append((a, r, c, t))
    events = np.asarray(events, dtype=np.int32)
    events = events[np.argsort(events[:, 3], kind="stable")]
    train = events[events[:, 3] <= 30]
    test_ev = events[events[:, 3] > 30]
    return events, train, test_ev


def _sgd(params, grads, lr):
    return jax.tree.map(lambda p, g: p - lr * g, params, grads)


def train_encoder(params, examples, log_pi, steps=40, lr=0.05, batch=32, seed=0):
    if not examples:
        return params
    rel = jnp.asarray(np.stack([e[0]["relation"] for e in examples]))
    direction = jnp.asarray(np.stack([e[0]["direction"] for e in examples]))
    bins = jnp.asarray(np.stack([e[0]["bins"] for e in examples]))
    mask = jnp.asarray(np.stack([e[0]["mask"] for e in examples]))
    target = jnp.asarray(np.array([e[1] for e in examples], dtype=np.int32))
    n = int(target.shape[0])
    key = jax.random.PRNGKey(seed)

    def loss_fn(p, idx):
        z = event_embedding(p, rel[idx], direction[idx], bins[idx])
        _, m = attention_pool(p, z, mask[idx])
        log_p = neural_log_probs(p, m, log_pi)
        return -jnp.mean(log_p[jnp.arange(idx.shape[0]), target[idx]])

    for s in range(steps):
        key, k = jax.random.split(key)
        idx = jax.random.choice(k, n, shape=(min(batch, n),), replace=False)
        loss, grads = jax.value_and_grad(loss_fn)(params, idx)
        params = _sgd(params, grads, lr)
        if s == 0 or (s + 1) % 20 == 0:
            print(f"step {s + 1} loss {float(loss):.4f}")
    return params


def score_split(params, events, queries, p_ct, token_n, log_pi, u_q=0.5, lam=1.0):
    p_ct_j = jnp.asarray(p_ct)
    token_n_j = jnp.asarray(token_n)
    log_pi_j = jnp.asarray(log_pi)
    gold = []
    filter_sets = []
    S_rows = []
    B_rows = []
    # Filter by other objects recorded for the same (s, r) anywhere.
    other = {}
    for s, r, o, _t in events:
        other.setdefault((int(s), int(r)), set()).add(int(o))
    for actor, rel, obj, t in queries:
        actor, rel, obj, t = int(actor), int(rel), int(obj), int(t)
        cand = np.arange(N_ENT, dtype=np.int32)
        tok = stack_candidates(events, actor, cand, t)
        p0 = jnp.asarray(random_scores(actor, rel, t, N_ENT))
        S, _A = log_scores(
            params,
            jnp.asarray(tok["relation"]),
            jnp.asarray(tok["direction"]),
            jnp.asarray(tok["bins"]),
            jnp.asarray(tok["mask"]),
            p_ct_j,
            token_n_j,
            log_pi_j,
            rel,
            p0,
            jnp.float32(u_q),
            lam=lam,
        )
        S_rows.append(np.array(S))
        B_rows.append(np.array(jnp.log(jnp.maximum(p0, 1e-12))))
        gold.append(obj)
        filter_sets.append(other.get((actor, rel), set()))
    S_rows = np.stack(S_rows)
    B_rows = np.stack(B_rows)
    return {
        "filtered_mrr": filtered_mrr(S_rows, gold, filter_sets),
        "backbone_filtered_mrr": filtered_mrr(B_rows, gold, filter_sets),
        "hits_at_1": hits_at_k(S_rows, gold, filter_sets, k=1),
        "n_test_queries": int(len(gold)),
    }


def main():
    rng = np.random.default_rng(0)
    events, train, test_ev = make_toy(rng)
    p_ct, token_n = fit_count_table(train, N_REL)
    pi = relation_prior(train, N_REL)
    log_pi = jnp.log(jnp.asarray(pi))
    examples = training_examples(train, N_REL)
    params = init_params(jax.random.PRNGKey(1), n_rel=N_REL)
    params = train_encoder(params, examples, log_pi)
    metrics = score_split(params, events, test_ev, p_ct, token_n, log_pi)
    out = {
        "filtered_mrr": metrics["filtered_mrr"],
        "backbone_filtered_mrr": metrics["backbone_filtered_mrr"],
        "hits_at_1": metrics["hits_at_1"],
        "n_entities": N_ENT,
        "n_relations": N_REL,
        "n_test_queries": metrics["n_test_queries"],
        "n_train_events": int(train.shape[0]),
        "n_train_examples": int(len(examples)),
    }
    path = Path("results.json")
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
