import torch
import pytest

from pharmacokinetics.state_space import NeuropharmacologyModel, PKPDParameters


def make_model(
    homeostasis_gain=0.0,
    dtype=torch.float64,
    hill=2.0,
    recovery=0.02,
    desensitization=0.1,
):
    return NeuropharmacologyModel(PKPDParameters(
        receptor_basis=torch.eye(3, 2, dtype=dtype),
        affinity=torch.ones(2, 1, dtype=dtype),
        ec50=torch.ones(2, dtype=dtype),
        hill=torch.full((2,), hill, dtype=dtype),
        emax=torch.tensor([1.0, 0.5], dtype=dtype),
        retention=torch.tensor([0.9], dtype=dtype),
        absorption=torch.tensor([1.0], dtype=dtype),
        recovery=torch.full((2,), recovery, dtype=dtype),
        desensitization=torch.full((2,), desensitization, dtype=dtype),
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


def test_high_dose_effect_has_diminishing_increment():
    model = make_model(recovery=0.0, desensitization=0.0)

    def effect(dose):
        step = model.step(
            model.initial_state(),
            torch.tensor([dose], dtype=torch.float64),
        )
        return float(step.receptor_effect[0])

    low_increment = effect(2.0) - effect(1.0)
    high_increment = effect(100.0) - effect(50.0)
    assert high_increment > 0
    assert high_increment < low_increment


@pytest.mark.parametrize(
    ("dtype", "dose"),
    [
        (torch.float16, 60_000.0),
        (torch.bfloat16, 1e30),
        (torch.float32, 1e30),
    ],
)
def test_hill_occupancy_is_finite_at_extreme_dose_and_hill(dtype, dose):
    model = make_model(dtype=dtype, hill=1_000.0)
    step = model.step(
        model.initial_state(),
        torch.tensor([dose], dtype=dtype),
    )
    assert torch.isfinite(step.occupancy).all()
    assert torch.isfinite(step.delta_h).all()
    assert torch.all((step.occupancy >= 0) & (step.occupancy <= 1))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_exact_zero_concentration_has_exact_zero_occupancy(dtype):
    model = make_model(dtype=dtype, hill=1_000.0)
    step = model.step(model.initial_state(), torch.zeros(1, dtype=dtype))
    assert torch.equal(step.occupancy, torch.zeros(2, dtype=dtype))
    assert torch.isfinite(step.occupancy).all()


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


def test_current_effect_uses_pre_exposure_receptor_state():
    model = make_model(recovery=0.0, desensitization=0.5)
    state = model.initial_state()
    step = model.step(state, torch.tensor([1.0], dtype=torch.float64))
    assert torch.equal(step.sensitivity_used, torch.ones(2, dtype=torch.float64))
    assert torch.allclose(step.receptor_effect, step.occupancy * model.p.emax)
    assert torch.all(step.state.sensitivity < step.sensitivity_used)


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


def test_no_homeostasis_means_no_negative_withdrawal_effect():
    model = make_model(homeostasis_gain=0.0)
    state = model.initial_state()
    for _ in range(40):
        state = model.step(state, torch.tensor([1.0], dtype=torch.float64)).state
    for _ in range(120):
        step = model.step(state, torch.zeros(1, dtype=torch.float64))
        state = step.state
        assert torch.all(step.receptor_effect >= 0)


def test_bolus_half_life_matches_retention_parameterization():
    half_life = 8
    retention = 2.0 ** (-1.0 / half_life)
    model = NeuropharmacologyModel(PKPDParameters(
        receptor_basis=torch.ones(1, 1, dtype=torch.float64),
        affinity=torch.ones(1, 1, dtype=torch.float64),
        ec50=torch.ones(1, dtype=torch.float64),
        hill=torch.ones(1, dtype=torch.float64),
        emax=torch.ones(1, dtype=torch.float64),
        retention=torch.tensor([retention], dtype=torch.float64),
        absorption=torch.ones(1, dtype=torch.float64),
        recovery=torch.zeros(1, dtype=torch.float64),
        desensitization=torch.zeros(1, dtype=torch.float64),
    ))
    state = model.initial_state()
    concentrations = []
    for token in range(half_life + 1):
        dose = torch.tensor([1.0 if token == 0 else 0.0], dtype=torch.float64)
        step = model.step(state, dose)
        state = step.state
        concentrations.append(float(state.concentration[0]))
    assert concentrations[half_life] == pytest.approx(
        0.5 * concentrations[0], rel=1e-12, abs=1e-12
    )


def test_parameter_identifiability_conventions_are_enforced():
    with pytest.raises(ValueError, match="unit L2 norm"):
        PKPDParameters(
            receptor_basis=torch.tensor([[2.0]], dtype=torch.float64),
            affinity=torch.ones(1, 1, dtype=torch.float64),
            ec50=torch.ones(1, dtype=torch.float64),
            hill=torch.ones(1, dtype=torch.float64),
            emax=torch.ones(1, dtype=torch.float64),
            retention=torch.zeros(1, dtype=torch.float64),
            absorption=torch.ones(1, dtype=torch.float64),
            recovery=torch.zeros(1, dtype=torch.float64),
            desensitization=torch.zeros(1, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="affinity row"):
        PKPDParameters(
            receptor_basis=torch.ones(1, 1, dtype=torch.float64),
            affinity=torch.tensor([[0.5]], dtype=torch.float64),
            ec50=torch.ones(1, dtype=torch.float64),
            hill=torch.ones(1, dtype=torch.float64),
            emax=torch.ones(1, dtype=torch.float64),
            retention=torch.zeros(1, dtype=torch.float64),
            absorption=torch.ones(1, dtype=torch.float64),
            recovery=torch.zeros(1, dtype=torch.float64),
            desensitization=torch.zeros(1, dtype=torch.float64),
        )


def test_nonfinite_and_unstable_parameters_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        PKPDParameters(
            receptor_basis=torch.ones(1, 1, dtype=torch.float64),
            affinity=torch.ones(1, 1, dtype=torch.float64),
            ec50=torch.tensor([float("nan")], dtype=torch.float64),
            hill=torch.ones(1, dtype=torch.float64),
            emax=torch.ones(1, dtype=torch.float64),
            retention=torch.zeros(1, dtype=torch.float64),
            absorption=torch.ones(1, dtype=torch.float64),
            recovery=torch.zeros(1, dtype=torch.float64),
            desensitization=torch.zeros(1, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="must be <= 1"):
        PKPDParameters(
            receptor_basis=torch.ones(1, 1, dtype=torch.float64),
            affinity=torch.ones(1, 1, dtype=torch.float64),
            ec50=torch.ones(1, dtype=torch.float64),
            hill=torch.ones(1, dtype=torch.float64),
            emax=torch.ones(1, dtype=torch.float64),
            retention=torch.zeros(1, dtype=torch.float64),
            absorption=torch.ones(1, dtype=torch.float64),
            recovery=torch.tensor([0.6], dtype=torch.float64),
            desensitization=torch.tensor([0.5], dtype=torch.float64),
        )
