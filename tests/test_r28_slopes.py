from experiments.r28_context_slopes_gpu import (
    binary_entropy_from_log_odds,
    pearson,
    summarize,
)


def test_entropy_is_highest_at_zero_margin():
    assert binary_entropy_from_log_odds(0.0) > binary_entropy_from_log_odds(5.0)


def test_pearson_detects_linear_relation():
    assert abs(pearson([1, 2, 3], [2, 4, 6]) - 1.0) < 1e-12


def test_summary_positive_fraction():
    result = summarize([-1.0, 1.0, 2.0, 3.0])
    assert result["positive_fraction"] == 0.75
