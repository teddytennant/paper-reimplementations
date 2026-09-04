# Importing the package turns on jax x64, which has to happen before any test
# builds an array. Selections are argmaxes over gains that can sit close together,
# and float32 matmuls run at TF32 on an H200, which is enough to reorder them.
import coverpruner  # noqa: F401
