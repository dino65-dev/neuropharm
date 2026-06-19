import torch

from administration.hooks import apply_additive_intervention


def test_zero_dose_exactly_preserves_logits():
    torch.manual_seed(0)
    resid = torch.randn(2, 3, 5)
    direction = torch.randn(5)
    readout = torch.randn(5, 7)
    baseline_logits = resid @ readout
    steered = apply_additive_intervention(resid, direction, coefficient=0.0)
    steered_logits = steered @ readout
    assert torch.equal(steered, resid)
    assert torch.equal(steered_logits, baseline_logits)


def test_explicit_token_position_only():
    resid = torch.zeros(1, 3, 4)
    direction = torch.ones(4)
    result = apply_additive_intervention(resid, direction, 2.0, token_index=-1)
    assert torch.equal(result[:, :-1, :], resid[:, :-1, :])
    assert torch.equal(result[:, -1, :], torch.full((1, 4), 2.0))

