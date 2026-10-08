# lea-ptq

Local error score and greedy mixed-precision allocator from arXiv 2610.09877.

Authors: Houache, Traonmilin, Aujol.

Install with `pip install -e .`.

Run tests with `JAX_PLATFORMS=cpu python -m pytest -q`.

Missing: no DRUNet, no diffusion, no images, no corruption protocol, no PSNR, rounding is JAX round not half-to-even, delta default is a user risk level not a table constant.
