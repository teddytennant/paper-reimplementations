# jax-cachebridge

JAX implementation of "CacheBridge: Efficient Cross-Model KV Cache Transfer"
(arXiv:2609.00891).

## install

uv pip install -e .

## run

python -m cachebridge.demo

Verified on CPU only. The model pair is a small synthetic GQA transformer whose target
KV states are planted from the source's, not Qwen3 or Ministral 3, so none of the
HellaSwag or retention numbers are reproduced. FUSED-FIT is the bounded chunked
construction written as a lax.scan rather than the paper's fused CUDA kernel, so it
matches the ridge solution and the eq 22 memory bound but not the 10.7x construction
speedup.
