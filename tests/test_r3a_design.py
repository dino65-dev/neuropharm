from collections import Counter

from experiments.r3a_design import DOMAINS, balanced_split, claims


def test_r3a_claims_and_split_are_balanced():
    rows = {row["claim_id"]: row for row in claims()}
    split = balanced_split()
    assert {name: len(ids) for name, ids in split.items()} == {
        "construction": 27, "validation": 9, "test": 18,
    }
    assert len(set().union(*map(set, split.values()))) == 54
    for name, expected in (("construction", 3), ("validation", 1), ("test", 2)):
        counts = Counter(rows[index]["domain"] for index in split[name])
        assert counts == Counter({domain: expected for domain in DOMAINS})
