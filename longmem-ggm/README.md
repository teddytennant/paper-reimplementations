tgg implements the direct Gaussian graphical model test for dependent Gaussian series from https://arxiv.org/abs/2609.30565: local-inverse residuals, an overlapping block bootstrap, and the consistency-empowered correction.

pip install -e .

JAX_PLATFORMS=cpu python -m pytest -q
JAX_PLATFORMS=cpu python demo.py

The sample covariance is Sigma_hat = (1/n) sum_t X_t X_t^T, the normalization in the methodology section, not the unbiased 1/(n-1) estimator.

The public test is Appendix Algorithm 1. The main-text statistic and that algorithm are the same unstudentized maximum residual. The appendix does not studentize, so this library does not add studentization. The consistency-empowered test is Appendix Algorithm 2: it pumps near-zero entries of the local inverse on the hypothesized support and reuses the direct-test critical value.

Cuts versus the paper: there is no Sherlock fMRI analysis. Tests use n=80 and p=12 rather than n=800 and p=2000, and the demo uses one modestly larger draw. The test generator uses a tridiagonal precision (1 on the diagonal, kappa on |i-j|=1) rather than the simulation section's bandwidth-4 band; simulation_precision implements that bandwidth-4 design. The two-sample graph estimator is a nodewise ridge precision followed by a thresholded partial correlation, a stand-in for the Chen, Xu, and Wu (2013) estimator cited in the two-sample corollary. The FFT generator applies the coefficient convolution channel-wise so that X_i = sum_t A_t eps_{i-t}. The appendix's flattened circulant reads only the first coordinate of each innovation block, which does not implement that product when the innovation dimension exceeds 1. Block length is an argument. The simulation section uses l = n^{2/3}. The default and the smoke call use floor(n^{1/3}). Tuning uses the natural logarithm in delta_n = sqrt(log n) and C_n = C_delta * sqrt(log p).
