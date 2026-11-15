# jax-metaquill-kip

JAX implementation of "Kernel Reboot: Breaking the Boundaries of Neural Tangent Kernels for Neural Fields" (arXiv:2609.03117).

## install

uv pip install -e .

## run

python -m metaquill_kip.demo --steps 80

NTK-KIP (Eq. 9), MetaQuill closed-form adaptation (Eq. 15) and the MetaQuill-KIP hybrid (Eqs. 9-11) on analytic 8x8 fields. Not a paper-for-paper reproduction of the MNIST / Flowers PSNR tables or the diffusion Functa pipeline.
