import torch

from pharmacokinetics.state_space import NeuropharmacologyModel, PKPDParameters


def make_model(homeostasis_gain=0.0):
    return NeuropharmacologyModel(PKPDParameters(
        receptor_basis=torch.eye(3, 2, dtype=torch.float64),
        affinity=torch.tensor([[1.0], [0.5]], dtype=torch.float64),
        ec50=torch.tensor([1.0, 1.0], dtype=torch.float64),
        hill=torch.tensor([2.0, 2.0], dtype=torch.float64),
        emax=torch.tensor([1.0, 0.5], dtype=torch.float64),
        elimination=torch.tensor([0.9], dtype=torch.float64),
        absorption=torch.tensor([1.0], dtype=torch.float64),
        recovery=torch.tensor([0.02, 0.02], dtype=torch.float64),
        desensitization=torch.tensor([0.1, 0.1], dtype=torch.float64),
        homeostasis_decay=0.9,
        homeostasis_gain=homeostasis_gain,
    ))


def test_shapes_and_zero_dose():
    model = make_model()
    step = model.step(model.initial_state(), torch.zeros(1, dtype=torch.float64))
    assert step.occupancy.shape == (2,)
    assert step.receptor_effect.shape == (2,)
    assert step.delta_h.shape == (3,)
    assert torch.equal(step.delta_h, torch.zeros(3, dtype=torch.float64))


def test_hill_occupancy_is_bounded_and_saturates():
    model = make_model()
    step = model.step(model.initial_state(), torch.tensor([1e9], dtype=torch.float64))
    assert torch.all(step.occupancy >= 0)
    assert torch.all(step.occupancy <= 1)
    assert torch.allclose(step.occupancy, torch.ones(2, dtype=torch.float64), atol=1e-12)


def test_concentration_respects_bounded_input_limit():
    model = make_model()
    state = model.initial_state()
    for _ in range(200):
        state = model.step(state, torch.tensor([1.0], dtype=torch.float64)).state
    assert state.concentration[0] <= 1.0 / (1.0 - 0.9) + 1e-8


def test_repeated_exposure_produces_tolerance():
    model = make_model()
    state = model.initial_state()
    effects = []
    for _ in range(80):
        step = model.step(state, torch.tensor([1.0], dtype=torch.float64))
        state = step.state
        effects.append(float(step.receptor_effect[0]))
    assert effects[-1] < effects[5]
    assert state.sensitivity[0] < 1.0


def test_withdrawal_can_produce_rebound_with_homeostasis():
    model = make_model(homeostasis_gain=0.08)
    state = model.initial_state()
    for _ in range(40):
        state = model.step(state, torch.tensor([1.0], dtype=torch.float64)).state
    effects = []
    for _ in range(120):
        step = model.step(state, torch.zeros(1, dtype=torch.float64))
        state = step.state
        effects.append(float(step.receptor_effect[0]))
    assert min(effects) < 0.0

