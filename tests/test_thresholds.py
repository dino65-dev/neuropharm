from evaluation.thresholds import directional_crossing, minimum_absolute_crossing


ROWS = [
    {"alpha": -12.0, "crossed": True},
    {"alpha": -2.0, "crossed": True},
    {"alpha": 0.0, "crossed": False},
    {"alpha": 4.0, "crossed": True},
]


def test_minimum_absolute_crossing_does_not_take_first_list_entry():
    assert minimum_absolute_crossing(ROWS, lambda row: row["crossed"]) == -2.0


def test_directional_crossings_are_separate():
    assert directional_crossing(ROWS, lambda row: row["crossed"], -1) == -2.0
    assert directional_crossing(ROWS, lambda row: row["crossed"], 1) == 4.0

