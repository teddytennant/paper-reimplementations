"""Short H200 smoke for scan-parallel latent jump training.

Twenty AdamW steps on one synthetic cyclic batch. Not a paper benchmark.
"""

import jax

from spvi.model import Config, init_params
from spvi.train import make_optimizer, synthetic_cyclic_batch, train_step


def main():
    devices = jax.devices()
    print("jax", jax.__version__, devices, flush=True)
    if not any(d.platform == "gpu" for d in devices):
        raise SystemExit("expected a GPU device")

    config = Config(l_max=8, n_quad=2, hidden=16, embed_dim=16)
    key = jax.random.PRNGKey(0)
    k_init, k_data, key = jax.random.split(key, 3)
    params = init_params(k_init, config)
    opt = make_optimizer(1e-3)
    opt_state = opt.init(params)
    batch = synthetic_cyclic_batch(
        k_data, batch_size=16, n_times=24, noise=0.05, config=config
    )
    losses = []
    for step in range(20):
        key, k_step = jax.random.split(key)
        params, opt_state, loss, _grads = train_step(
            params, opt_state, opt, batch, k_step, config
        )
        value = float(loss)
        losses.append(value)
        print(f"step {step} loss {value:.6f}", flush=True)
    print(
        "first",
        losses[0],
        "last",
        losses[-1],
        "decreased",
        losses[-1] < losses[0],
        flush=True,
    )


if __name__ == "__main__":
    main()
