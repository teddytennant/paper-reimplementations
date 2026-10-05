DAGS is disentangled appearance-and-geometry steering of a frozen image DiT (arXiv 2610.02567).

pip install -e .

pytest -q

The injection, temporal guidance, demodulation, and lighting recurrence match the paper. Encoder widths, the U-Net, the blur kernel, and facetization are not specified, so those are thin stand-ins or omitted. Flux.2-Klein and the path-traced dataset are not included, and the reported PSNR numbers are not reproduced. An H200 smoke of train.py (job 753778) dropped stabilizer loss from 0.292 to 0.00893 in 30 steps.
