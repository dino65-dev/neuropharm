from collections import Counter

from experiments.r26_design import (
    DOMAINS,
    QUESTION_TEMPLATES,
    RESPONSE_FRAMES,
    balanced_claim_split,
    claims,
)


def test_r26_factorial_design_has_expected_size():
    assert len(claims()) == 72
    assert len(QUESTION_TEMPLATES) == 4
    assert len(RESPONSE_FRAMES) == 8
    assert len(claims()) * len(QUESTION_TEMPLATES) * len(RESPONSE_FRAMES) * 2 == 4608


def test_claim_split_is_balanced_and_disjoint():
    rows = {int(row["claim_id"]): row for row in claims()}
    split = balanced_claim_split(20260621)
    assert {name: len(ids) for name, ids in split.items()} == {
        "construction": 36,
        "validation": 18,
        "test": 18,
    }
    assert len(set().union(*map(set, split.values()))) == 72
    for name, expected in (("construction", 4), ("validation", 2), ("test", 2)):
        counts = Counter(str(rows[index]["domain"]) for index in split[name])
        assert counts == Counter({domain: expected for domain in DOMAINS})
