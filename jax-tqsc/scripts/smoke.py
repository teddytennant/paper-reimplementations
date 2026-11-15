import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

from jax_tqsc.data import make_synthetic_dataset
from jax_tqsc.model import count_params, init_tqsc
from jax_tqsc.train import evaluate, train_tqsc


def main() -> None:
    train = make_synthetic_dataset(192, seed=0)
    test = make_synthetic_dataset(64, seed=1)
    key = jax.random.PRNGKey(0)
    params0 = init_tqsc(key)
    n_params = count_params(params0)
    init_train = evaluate(params0, train["M"], train["rho"])
    init_test = evaluate(params0, test["M"], test["rho"])
    params, log = train_tqsc(
        jax.random.PRNGKey(1),
        train["M"],
        train["rho"],
        steps=120,
        batch_size=32,
        lr=1e-3,
        params=params0,
    )
    after_train = evaluate(params, train["M"], train["rho"])
    after_test = evaluate(params, test["M"], test["rho"])
    print(f"params: {n_params}")
    print(f"init  train_fid={init_train['fidelity']:.4f} test_fid={init_test['fidelity']:.4f} mse={init_train['mse']:.4f}")
    print(f"after train_fid={after_train['fidelity']:.4f} test_fid={after_test['fidelity']:.4f} mse={after_train['mse']:.4f}")
    print(f"mixed baseline test_fid={after_test['mixed_fidelity']:.4f}")
    print(f"loss {log['loss'][0]:.4f} -> {log['loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
