import jax

# Gaps, value residuals and unbiasedness checks compare small differences of
# numbers computed through matmuls. On an NVIDIA device those run at TF32 in
# float32 and lose roughly three decimal digits, so everything here runs in
# float64.
jax.config.update("jax_enable_x64", True)
