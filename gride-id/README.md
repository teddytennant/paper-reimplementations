JAX kernel of the componentwise Gride estimator (arXiv:2609.37114): closed-form KL for generic-order distance ratios, plus Full and Profiled angular calibration.

PYTHONPATH=src python -m pytest
PYTHONPATH=src python train.py

Does not reproduce the 24-manifold table or the CIFAR and ImageNet layer curves. Those need the authors' reference samples and image embeddings. This checks the closed-form KL identity, the finite-sum I(gamma) against quadrature, the TWO-NN special case of the MLE, and that a synthetic low-dimensional cloud is not estimated as ambient dimension.

Neighbor orders are k and 2k. Concentration uses the Fisher piecewise approximation, which the paper names and does not write: if R < 0.53 then tau = 2R + R^3 + 5 R^5 / 6, if R < 0.85 then tau = -0.4 + 1.39 R + 0.43 / (1 - R), otherwise tau = 1 / (R^3 - 4 R^2 + 3 R). R is the mean resultant length, clipped to [0, 1-1e-8] before the last piece, and tau is 0 when R is within 1e-8 of 0. Angular KL is the von Mises KL. Profiled sets the reference mean direction equal to the data direction. Reference angles are two-NN angles of uniform unit-ball samples.
