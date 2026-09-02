# jax-diffusion-is-attention

JAX implementation of "Denoising Diffusion Generative Models Secretly
Calculate Attentions" (arXiv:2609.00885).

## install

uv pip install -e .

## run

python -m diffusion_is_attention

Docstrings name the equation each function implements, so read them next to the
paper. What is not here: no trained VAE, so Algorithm 1's codec defaults to the
identity and everything runs on synthetic latents. `scripts/train_mnist_ae.py`
trains a small conv autoencoder if you have MNIST on disk, and no test touches
it. No MNIST, Fashion-MNIST, FFHQ or CelebA numbers and no FID. The Sec. IV tree
is built top down rather than by the online insertion the paper describes, since
the O(log |D|) claim needs a bounded branching factor. The Figure 2 coordinates
are digitized off the PDF, and the trajectory they give ends at s2 rather than
s1 as published; there is a note about that in `convergence.py`.
