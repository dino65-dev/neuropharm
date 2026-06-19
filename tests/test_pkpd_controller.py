import torch

from administration.pkpd_controller import (
    PKPDGenerationController,
    protocol_schedule,
)
from pharmacokinetics.state_space import NeuropharmacologyModel, PKPDParameters


def make_controller(protocol="bolus", dose=1.0):
    model = NeuropharmacologyModel(PKPDParameters(
        receptor_basis=torch.tensor([[1.0], [0.0]], dtype=torch.float64),
        affinity=torch.ones(1, 1, dtype=torch.float64),
        ec50=torch.tensor([0.5], dtype=torch.float64),
        hill=torch.tensor([2.0], dtype=torch.float64),
        emax=torch.ones(1, dtype=torch.float64),
        retention=torch.tensor([2.0 ** (-1.0 / 8.0)], dtype=torch.float64),
        absorption=torch.ones(1, dtype=torch.float64),
        recovery=torch.zeros(1, dtype=torch.float64),
        desensitization=torch.zeros(1, dtype=torch.float64),
        homeostasis_gain=0.0,
    ))
    return PKPDGenerationController(
        model=model,
        dose_schedule=protocol_schedule(protocol, dose),
    )


def test_prefill_advances_once_and_modifies_only_final_position():
    controller = make_controller()
    resid = torch.zeros(1, 17, 2, dtype=torch.float64)
    result = controller.hook(resid)
    assert controller.step_index == 1
    assert len(controller.trace) == 1
    assert controller.trace[0]["pkpd_step_index"] == 0
    assert controller.trace[0]["is_prefill"] is True
    assert torch.equal(result[:, :-1, :], resid[:, :-1, :])
    assert not torch.equal(result[:, -1, :], resid[:, -1, :])


def test_cached_decode_call_advances_one_additional_step():
    controller = make_controller()
    controller.hook(torch.zeros(1, 17, 2, dtype=torch.float64))
    controller.hook(torch.zeros(1, 1, 2, dtype=torch.float64))
    assert controller.step_index == 2
    assert [row["pkpd_step_index"] for row in controller.trace] == [0, 1]
    assert controller.trace[1]["forward_sequence_length"] == 1
    assert controller.trace[1]["is_prefill"] is False


def test_zero_dose_returns_residual_exactly_unchanged():
    controller = make_controller(protocol="zero", dose=1.0)
    resid = torch.randn(1, 9, 2, dtype=torch.float64)
    result = controller.hook(resid)
    assert result is resid
    assert torch.equal(result, resid)
    assert controller.trace[0]["delta_h_norm"] == 0.0


def test_pulse_schedule_uses_explicit_step_index():
    schedule = protocol_schedule("pulses", dose=2.0, stop=20, pulse_period=8)
    values = [float(schedule(index)[0]) for index in range(21)]
    assert [index for index, value in enumerate(values) if value > 0] == [0, 8, 16]

