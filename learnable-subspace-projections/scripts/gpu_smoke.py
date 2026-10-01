"""One short subspace fit on whatever device JAX sees. Exit 1 if the loss does not fall."""

import jax
from learnable_subspace.train import fit_mlp, init_mlp

print("jax", jax.__version__, jax.devices())
model, x = init_mlp(jax.random.PRNGKey(0), width=32, k=8, batch=64)
_model, losses = fit_mlp(model, x, steps=40, lr=1e-2)
start = float(losses[0])
end = float(losses[-1])
print("loss_start", start)
print("loss_end", end)
if not (end < start):
    raise SystemExit(1)
print("lsp gpu smoke ok")
