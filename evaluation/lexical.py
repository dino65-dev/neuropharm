from __future__ import annotations

import re
from collections.abc import Iterable


def count_boundary_hits(text: str, terms: Iterable[str]) -> int:
    """Count terms with word/phrase boundaries, at most once per term."""

    normalized = text.casefold()
    return sum(
        bool(re.search(rf"(?<!\w){re.escape(term.casefold())}(?!\w)", normalized))
        for term in terms
    )

