import torch

from experiments.r27_nuisance_optimization_gpu import (
    attenuated_direction,
    empirical_difficulty,
    pareto_front,
)


def test_attenuation_endpoints_match_raw_and_projected():
    receptor = torch.tensor([1.0, 1.0, 0.0])
    receptor = receptor / receptor.norm()
    nuisance = torch.tensor([[1.0], [0.0], [0.0]])
    raw = attenuated_direction(receptor, nuisance, 0.0)
    projected = attenuated_direction(receptor, nuisance, 1.0)
    assert torch.allclose(raw, receptor)
    assert torch.allclose(projected, torch.tensor([0.0, 1.0, 0.0]))


def test_empirical_difficulty_has_all_three_strata():
    result = empirical_difficulty([0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
    assert set(result) == {"hard", "medium", "easy"}


def test_pareto_front_removes_dominated_row():
    base = {
        "definite_answer_delta": 1.0,
        "numeric_confidence_delta": 1.0,
        "answer_margin_delta": 1.0,
        "language_delta": 1.0,
        "ambiguous_definite_delta": 1.0,
        "brier_delta": 1.0,
        "nll_delta": 1.0,
    }
    better = dict(base)
    better["language_delta"] = 0.5
    assert pareto_front({"better": better, "worse": base}) == ["better"]
