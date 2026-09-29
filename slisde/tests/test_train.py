"""train.py smoke: calibration loss decreases on the synthetic task (Eq. 5)."""

from train import run_smoke


def test_train_loss_decreases():
    start, end = run_smoke(steps=25, batch=48, n_steps=6, print_loss=False)
    assert start > 0.0
    assert end < start
