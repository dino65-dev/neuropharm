import torch

from experiments.r26_analyze_crossed import (
    effect_partition,
    heldout_r2,
    principal_stability,
    ridge_classifier_accuracy,
)


def test_balanced_effect_partition_separates_response_interaction():
    contrasts = torch.zeros(3, 2, 4, 5)
    contrasts[..., 0] = 2.0
    contrasts[:, :, 0, 1] = 3.0
    contrasts[:, :, 1, 1] = -3.0
    result = effect_partition(contrasts, ["a", "a", "b"])
    fractions = result["fraction_of_contrast_energy"]
    assert fractions["certainty_main"] > 0
    assert fractions["certainty_x_response"] > 0
    assert fractions["certainty_x_question"] == 0


def test_reconstruction_and_principal_stability():
    basis = torch.eye(5)[:, :2]
    values = torch.tensor([[1.0, 2.0, 0, 0, 0], [-1.0, -2.0, 0, 0, 0]])
    assert heldout_r2(values, torch.zeros(5), basis) == 1.0
    stability = principal_stability(basis, basis)
    assert stability["mean_cosine_squared"] == 1.0
    assert stability["minimum_cosine"] == 1.0


def test_ridge_classifier_generalizes_simple_classes():
    train_x = torch.tensor([[-2.0], [-1.0], [1.0], [2.0]])
    train_y = torch.tensor([0, 0, 1, 1])
    assert ridge_classifier_accuracy(
        train_x, train_y, train_x, train_y, classes=2
    ) == 1.0
