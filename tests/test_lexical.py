from evaluation.lexical import count_boundary_hits


def test_single_character_term_does_not_match_inside_word():
    assert count_boundary_hits("use this result", ["u"]) == 0
    assert count_boundary_hits("u can do this", ["u"]) == 1


def test_phrase_boundaries():
    assert count_boundary_hits("There is no doubt.", ["no doubt"]) == 1
    assert count_boundary_hits("There is no doubtful result.", ["no doubt"]) == 0

