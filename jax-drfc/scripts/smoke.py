#!/usr/bin/env python3
"""Short CPU simulation of the cancellation toy. Prints S_loc, S_dec, DRFC switches."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax

from jax_drfc.drfc import DRFCConfig, run_drfc
from jax_drfc.env import count_s_loc_s_dec, make_cancellation_env


def main() -> None:
    horizon = 96
    env = make_cancellation_env(
        horizon=horizon,
        n_agents=2,
        n_arms=2,
        delta=0.25,
        eps=0.20,
        period=16,
        switch_time=None,
        noise="none",
    )
    s_loc, s_dec = count_s_loc_s_dec(env.mean_path)
    cfg = DRFCConfig(
        n_agents=2,
        n_arms=2,
        horizon=horizon,
        h=2,
        M=24,
        Hscan=64,
        delta=0.1,
        C=0.15,
        init_candidate=0,
    )
    result = run_drfc(env, cfg, jax.random.PRNGKey(0))
    print(f"S_loc={s_loc} S_dec={s_dec} DRFC_switches={result.n_switches}")

    env_sw = make_cancellation_env(
        horizon=horizon,
        n_agents=2,
        n_arms=2,
        delta=0.25,
        eps=0.20,
        period=16,
        switch_time=48,
        noise="none",
    )
    s_loc2, s_dec2 = count_s_loc_s_dec(env_sw.mean_path)
    result2 = run_drfc(env_sw, cfg, jax.random.PRNGKey(0))
    print(
        f"decision_switch S_loc={s_loc2} S_dec={s_dec2} "
        f"DRFC_switches={result2.n_switches}"
    )


if __name__ == "__main__":
    main()
