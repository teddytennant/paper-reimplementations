# jax-mkmeans-dpd

JAX implementation of "Robust K-means Clustering using the Density Power
Divergence Measure" (arXiv:2608.30093).

## install

uv pip install -e .

## run

python -m mkmeans_dpd.demo

The Covid-19 case-fatality analysis of Section 5.2 is not here, and neither are
the DBSCAN and GMM rows of Tables 2 and 3. Step 5's fallback line search walks
down the steepest-descent direction rather than a BFGS one; it still satisfies
the non-increase requirement the proof of Proposition 3.1 uses, but it is not
what the paper specifies. Three things the paper leaves unstated I had to pick:
the eigenvalue floor delta of Theorem 3.2, set to 1e-3 times the mean eigenvalue
of the global covariance, without which Q is unbounded below and a cluster
collapses onto p + 1 points; the trimming level for CHI_trim, 0.1; and how the
30 outliers of Section 4.2 were drawn, here uniform over a padded bounding box
and outside every group's 99% ellipsoid. Accuracy comes out near Tables 2 and 3,
0.99 and 0.97 against their 0.955 and 0.967, on different iteration counts.
