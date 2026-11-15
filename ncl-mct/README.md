JAX kernel of the NCL-MCT integrator (arXiv:2609.37113): mass and compression factors stepped by a shared split scheme, with known-law supervision of mobility and force.

PYTHONPATH=src python -m pytest
PYTHONPATH=src python train.py

Does not reproduce the seven PDE tables. Those use a convolutional neural operator on 128 by 128 fields and PhysicsNeMo. This checks the 1D periodic split step (finite-volume compression, semi-Lagrangian mass), the known-law versus velocity identifiability gap, and a short fit of a local constitutive map on linear diffusion.

Reaction half-steps follow Equation 19. Passing no rate map makes those half-steps the identity, and the tests plus train.py stay on that nonreactive path. Inside one step the velocity is evaluated from the current density and then held fixed, so the constitutive map is not called again during the compression or semi-Lagrangian substeps. Reinitialization, when a period is set, resets the compression factor to 1 and the mass factor to the reconstructed density after the step, then starts the period over. The curl penalty is a periodic central-difference check on a 2D force and is identically zero for a 1D force. No convolutional neural operator is included. The integrator does not claim exact discrete mass conservation of the reconstructed density.
