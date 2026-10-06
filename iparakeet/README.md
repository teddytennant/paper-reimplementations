Integer relative-attention score fusion and a minimax Swish approximation, from https://arxiv.org/abs/2609.30846.

JAX_PLATFORMS=cpu python -m pytest -q

JAX_PLATFORMS=cpu python demo.py

No 0.6B Parakeet, no LibriSpeech, no NPU, no WER or real-time-factor claim. Softmax is float, not the on-device I-BERT kernel. The Swish coefficients are the paper's a=-0.1240, c=2.4632.

Eq. 10 floors the real-valued fused sum (`jnp.floor`). Eq. 11 rounds the dyadic multipliers with `round` (half to even) and arithmetic-right-shifts by n=16 with round-half-away-from-zero. Phi runs on the integer position scores, before dequantization. The dyadic accumulator is int32.
