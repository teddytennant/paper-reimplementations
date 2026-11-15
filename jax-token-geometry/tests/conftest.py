"""Shared fixtures. Everything expensive is built once per session."""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

from tokengeom import geometry as G  # noqa: E402
from tokengeom import model as M  # noqa: E402
from tokengeom import recurrent as R  # noqa: E402
from tokengeom import train as T  # noqa: E402
from tokengeom.generative import SemanticModel, SimulationConfig  # noqa: E402

# Appendix B.2 trains for up to 3e5 updates at 3e-4. The suite has a couple of
# thousand, so the learning rate is raised to land in a comparable epsilon_p.
ENCODER_STEPS = 1200
ENCODER_DEPTH = 8
ENCODER_LR = 3e-3
FIT_STEPS = (25, 40, 60, 100, 250, 1000, 4000)


def encoder_config(sim, depth=ENCODER_DEPTH, n_sub_layers=1):
    return M.EncoderConfig(
        vocab_size=sim.config.vocab_size,
        width=8,
        n_heads=2,
        ffn_width=32,
        n_sub_layers=n_sub_layers,
        depth=depth,
    )


@pytest.fixture(scope="session")
def sim():
    return SemanticModel(SimulationConfig.small(seed=0))


@pytest.fixture(scope="session")
def semantic_enum(sim):
    return T.enumerate_semantic_contexts(sim)


@pytest.fixture(scope="session")
def token_enum(sim):
    return T.enumerate_contexts(sim)


@pytest.fixture(scope="session")
def fit_path(sim, semantic_enum):
    """Unrestricted fits along a training path, ordered by decreasing epsilon_p.

    Figures 2, 4 and 5 all plot against epsilon_p along such a path.
    """
    out = []
    for steps in FIT_STEPS:
        params, _ = T.fit_unrestricted(sim, semantic_enum, width=8, n_steps=steps)
        out.append(G.FittedPair.from_unrestricted(params, semantic_enum))
    return out


@pytest.fixture(scope="session")
def fitted(fit_path):
    """The best-fitting unrestricted pair, the small-epsilon_p regime."""
    return min(fit_path, key=lambda pair: pair.epsilon_p)


@pytest.fixture(scope="session")
def curvature(fitted):
    """(b_Theta^2, B_Theta^2) of Assumption 1.3 for the fitted pair."""
    return fitted.curvature_bounds()


@pytest.fixture(scope="session")
def encoder(sim, token_enum):
    """A recurrent-depth transformer encoder trained on sampled pairs."""
    config = encoder_config(sim)
    params, _ = T.train(
        sim,
        config,
        seed=0,
        n_steps=ENCODER_STEPS,
        n_train=200_000,
        batch_size=256,
        learning_rate=ENCODER_LR,
    )
    pair = G.FittedPair.from_encoder(params, token_enum, config)
    return params, config, pair


@pytest.fixture(scope="session")
def ideal(encoder):
    """Z_j(X), the decoder-optimal representation for the encoder's head."""
    _, _, pair = encoder
    return R.ideal_representation(pair, n_steps=2000)
