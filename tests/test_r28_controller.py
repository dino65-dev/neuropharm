import torch

from experiments.r28_gain_controller_gpu import fit_ridge, predict


def test_ridge_controller_predicts_linear_slope():
    x = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float64)
    y = torch.tensor([1.0, 3.0, 5.0, 7.0], dtype=torch.float64)
    model = fit_ridge(x, y, ridge=1e-9)
    prediction = predict(model, x)
    assert torch.max(torch.abs(prediction - y)) < 1e-7
