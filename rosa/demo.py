"""Print the rotating active sets and a finite synthetic loss."""

import jax
import jax.numpy as jnp

from rosa import RosaTrainer, active_set, init_affine_stack, partition_layers


def _fmt(ids):
    def sort_key(layer_id):
        if isinstance(layer_id, int):
            return (0, layer_id, "")
        return (1, 0, str(layer_id))

    return "{" + ", ".join(str(layer_id) for layer_id in sorted(ids, key=sort_key)) + "}"


def main():
    layers = list(range(1, 9))
    schedule = partition_layers(layers, frozen=(1, 2), anchor=(8,), group_size=2)
    print("toy R", list(schedule.rotatable), "K", schedule.K)
    print("groups", schedule.groups)
    for epoch in range(1, 5):
        plan = active_set(schedule, epoch)
        print(
            f"epoch {epoch} k {plan.k} S {_fmt(plan.active)} l_start {plan.l_start}"
        )

    llama_layers = ["embedding", *range(32), "head"]
    llama = partition_layers(
        llama_layers,
        frozen=["embedding", *range(8)],
        anchor=[31, "head"],
        group_size=8,
    )
    print("llama groups", llama.groups, "K", llama.K)
    for epoch in range(1, 5):
        plan = active_set(llama, epoch)
        print(
            f"llama epoch {epoch} k {plan.k} S {_fmt(plan.active)} l_start {plan.l_start}"
        )

    key = jax.random.PRNGKey(0)
    k_params, k_x, k_y = jax.random.split(key, 3)
    params = init_affine_stack(8, 4, k_params)
    x = jax.random.normal(k_x, (16, 4), dtype=jnp.float32)
    y = 0.1 * jax.random.normal(k_y, (16, 4), dtype=jnp.float32)
    trainer = RosaTrainer(params, schedule, rho=0.2, delta_t=600, learning_rate=1e-2)
    losses = []
    for epoch in range(1, 5):
        trainer.begin_epoch(epoch)
        for _ in range(3):
            losses.append(trainer.step(x, y))
    loss = losses[-1]
    print(f"final loss {loss}")
    if not (loss == loss and abs(loss) != float("inf")):
        raise SystemExit("loss was not finite")


if __name__ == "__main__":
    main()
