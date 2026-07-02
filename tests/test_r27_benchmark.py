from collections import Counter

from experiments.r27_benchmark import (
    benchmark_split,
    build_calibration_items,
)


def test_r27_benchmark_size_and_composition():
    rows = build_calibration_items()
    assert len(rows) == 360
    labels = Counter(row["label"] for row in rows)
    assert labels == Counter({1: 144, 0: 144, None: 72})


def test_r27_benchmark_split_is_disjoint_and_complete():
    split = benchmark_split()
    validation = set(split["validation"])
    test = set(split["test"])
    assert not validation & test
    assert len(validation | test) == 360
    assert len(validation) == len(test) == 180
