from experiments.r26_causal_and_dose_gpu import factual_metrics, fit_curve


def test_factual_metrics_reward_correct_margin_signs():
    result = factual_metrics([3.0, -3.0], [1, 0])
    assert result["accuracy"] == 1.0
    assert result["brier"] < 0.01


def test_curve_fit_recovers_linear_as_best_or_tied():
    doses = [0.0, 0.25, 0.5, 1.0]
    values = [1.0 + 2.0 * dose for dose in doses]
    result = fit_curve(doses, values)
    assert result["models"]["linear"]["sse"] < 1e-20
    assert result["models"]["biphasic"]["parameters"]["gamma"] >= 0
