import torch

from experiments.r25_subspace_reproducibility import (
    fit_centered_basis,
    heldout_r2,
    marginal_eta_squared,
)


def test_heldout_reconstruction_recovers_known_subspace():
    train = torch.tensor([
        [1.0, 0.0, 5.0],
        [-1.0, 0.0, 5.0],
        [0.0, 1.0, 5.0],
        [0.0, -1.0, 5.0],
    ])
    test = torch.tensor([
        [2.0, 1.0, 5.0],
        [-2.0, -1.0, 5.0],
    ])
    mean, basis = fit_centered_basis(train, 2)
    assert heldout_r2(test, mean, basis) > 0.999999


def test_marginal_eta_squared_detects_group_axis():
    coordinate = torch.tensor([-1.0, -1.0, 1.0, 1.0])
    assert marginal_eta_squared(coordinate, ["a", "a", "b", "b"]) == 1.0
    assert marginal_eta_squared(coordinate, ["a", "b", "a", "b"]) == 0.0

