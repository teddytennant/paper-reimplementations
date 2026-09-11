"""Small synthetic train/eval of the eq. (17) estimator (CPU, minutes)."""

from __future__ import annotations

import json
from pathlib import Path

import jax

from puauc_conf.data import generate_biased_pu, generate_labeled_test
from puauc_conf.model import init_linear
from puauc_conf.train import evaluate_auc, fit_labeling_classifier, predict_u, train_score


def main() -> None:
    key = jax.random.PRNGKey(0)
    k_tr, k_te, k_u, k_s, k_fit = jax.random.split(key, 5)
    data = generate_biased_pu(k_tr, n=300, pi=0.35)
    x_te, y_te = generate_labeled_test(k_te, n=400, pi=0.35)

    u_params = fit_labeling_classifier(
        data.x_labeled, data.x_unlabeled, steps=150, lr=0.2, key=k_u
    )
    u_hat = predict_u(u_params, data.x_labeled)
    params = init_linear(k_s, dim=2, scale=0.1)
    params, losses = train_score(
        data.x_labeled,
        data.r,
        data.x_unlabeled,
        u_hat,
        params,
        steps=250,
        lr=0.08,
        key=k_fit,
    )
    test_auc = evaluate_auc(params, x_te, y_te)
    out = {
        "train_loss_start": losses[0],
        "train_loss_end": losses[-1],
        "test_auc": test_auc,
        "n_labeled": int(data.x_labeled.shape[0]),
        "n_unlabeled": int(data.x_unlabeled.shape[0]),
        "steps": 250,
    }
    print(json.dumps(out, indent=2))
    Path("results.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
