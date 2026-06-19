from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping


Row = Mapping[str, float]


def minimum_absolute_crossing(
    rows: Iterable[Row],
    predicate: Callable[[Row], bool],
) -> float | None:
    """Return the threshold crossing nearest zero."""
    crossings = [float(row["alpha"]) for row in rows if predicate(row)]
    return min(crossings, key=lambda alpha: (abs(alpha), alpha)) if crossings else None


def directional_crossing(
    rows: Iterable[Row],
    predicate: Callable[[Row], bool],
    sign: int,
) -> float | None:
    """Return the nearest-zero crossing on one signed branch."""
    if sign not in {-1, 1}:
        raise ValueError(f"sign must be -1 or +1, got {sign}")
    crossings = [
        float(row["alpha"])
        for row in rows
        if sign * float(row["alpha"]) > 0 and predicate(row)
    ]
    return min(crossings, key=abs) if crossings else None

