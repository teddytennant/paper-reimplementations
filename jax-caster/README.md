# jax-caster

JAX implementation of "Adapting Without Gradients: Affine Statistics Transport
and What Its Certificate Can Tell You" (arXiv:2609.00374).

## install

uv pip install -e .

## run

python -m caster.demo

This works on synthetic Gaussian features, not the four backbones and seven
datasets of the paper, so none of the reported accuracy numbers are reproduced.
There is no backbone here at all, so the gate falls back to the source Gaussian
discriminant unless you pass it head predictions. Of the fixed thresholds, only
tau = 0.8 is the paper's; lambda, beta, the confidence threshold and the two
coverage requirements are described as globally fixed but never printed, and
the released PyTorch code does not include the method module, so those are my
choices. The authors' original implementation is at
https://github.com/salimkhazem/Caster.
