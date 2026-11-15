"""Tiny synthetic train + online Laplace update. CPU-ok."""

import jax
import jax.numpy as jnp

from gvbll.data import two_community_graph
from gvbll.online import online_laplace_update, posterior_from_vbll
from gvbll.train import evaluate, init_gvbll, train
from gvbll.vbll import predict_map


def main():
    key = jax.random.PRNGKey(0)
    k_data, k_init, k_train = jax.random.split(key, 3)
    x, adj, y = two_community_graph(k_data, n_per=12, feat_dim=6, noise=0.1)
    params = init_gvbll(
        k_init, in_dim=x.shape[1], n_classes=2, hidden_dim=16, embed_dim=8
    )
    before = evaluate(params, x, adj, y)
    params, _losses = train(
        params,
        x,
        adj,
        y,
        k_train,
        steps=80,
        lr=2e-2,
        n_samples=6,
        anneal_steps=25,
    )
    after = evaluate(params, x, adj, y)
    print(f"loss_before={float(before['nll']):.6f}")
    print(f"loss_after={float(after['nll']):.6f}")
    print(f"acc={float(after['acc']):.4f}")

    post_train = posterior_from_vbll(params["vbll"])
    h = after["h"]
    # Flip the label of node 0 and take one online step.
    y_new = 1 - int(y[0])
    p_before = float(predict_map(post_train["mean"], h[0:1])[0, y_new])
    post = online_laplace_update(
        post_train,
        h[0],
        jnp.array(y_new),
        post_train,
        lambda_=0.9,
        eta=1.0,
        beta=0.1,
    )
    p_after = float(predict_map(post["mean"], h[0:1])[0, y_new])
    print(f"online_p_before={p_before:.4f} online_p_after={p_after:.4f}")


if __name__ == "__main__":
    main()
