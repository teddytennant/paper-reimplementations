# jax-maskcode

JAX implementation of "MaskCode: Mask Transformer for Feedback-Assisted Coding With Linear Block Codes"
(arXiv:2609.00715).

## install

uv pip install -e .

## run

python -m maskcode.train --steps 40 --batch 32

Inner feedback Transformer with a Tanner-graph attention mask and a soft-syndrome input, trained on a tiny Hamming outer code. Not a reproduction of the BCH/LDPC BLER curves or the 1.5 dB SNR claim.
