from experiments.r29_externality_panel_gpu import generated_metrics


def test_generation_metrics_detect_repetition():
    repetitive = generated_metrics(["a b c d a b c d a b c d"])
    diverse = generated_metrics(["a b c d e f g h i j k l"])
    assert (
        repetitive["mean_fourgram_repetition"]
        > diverse["mean_fourgram_repetition"]
    )
    assert (
        repetitive["mean_lexical_diversity"]
        < diverse["mean_lexical_diversity"]
    )
