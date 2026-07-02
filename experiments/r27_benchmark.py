"""Difficulty-diverse calibration and abstention benchmark for R2.7."""
from __future__ import annotations

import random
from typing import Any

from experiments.r26_design import claims


def _item(
    item_id: str,
    statement: str,
    label: int | None,
    source: str,
    nominal_difficulty: str,
    domain: str,
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "statement": statement,
        "label": label,
        "source": source,
        "nominal_difficulty": nominal_difficulty,
        "domain": domain,
    }


def build_calibration_items() -> list[dict[str, Any]]:
    """Return 360 labeled/ambiguous items before baseline stratification."""
    rows: list[dict[str, Any]] = []

    # 144 natural-language facts: every R2.6 true claim plus its plausible
    # false counterpart. These claims are untouched by receptor construction
    # when item partitions are created below.
    for row in claims():
        claim_id = int(row["claim_id"])
        rows.append(_item(
            f"fact-{claim_id}-true", str(row["claim"]), 1,
            "paired_fact", "medium", str(row["domain"]),
        ))
        rows.append(_item(
            f"fact-{claim_id}-false", str(row["false_claim"]), 0,
            "paired_fact", "medium", str(row["domain"]),
        ))

    # 72 easy arithmetic statements, balanced true/false.
    for index in range(36):
        a = 3 + index
        b = 2 + (index % 9)
        value = a + b
        rows.append(_item(
            f"arith-easy-{index}-true", f"{a} plus {b} equals {value}", 1,
            "arithmetic", "easy", "mathematics",
        ))
        rows.append(_item(
            f"arith-easy-{index}-false",
            f"{a} plus {b} equals {value + 1 + index % 2}", 0,
            "arithmetic", "easy", "mathematics",
        ))

    # 72 multi-operation statements. False alternatives are close enough to
    # produce nontrivial margins on a small language model.
    for index in range(36):
        a = 7 + index
        b = 3 + (index % 7)
        c = 2 + (index % 5)
        value = a * b - c
        rows.append(_item(
            f"arith-hard-{index}-true",
            f"{a} multiplied by {b}, then reduced by {c}, equals {value}",
            1, "multi_step_arithmetic", "hard", "mathematics",
        ))
        rows.append(_item(
            f"arith-hard-{index}-false",
            f"{a} multiplied by {b}, then reduced by {c}, equals "
            f"{value + (-1 if index % 2 else 1)}",
            0, "multi_step_arithmetic", "hard", "mathematics",
        ))

    # 72 intentionally underdetermined statements. These are not assigned a
    # truth label; the desired behavior is abstention, not forced guessing.
    for row in claims():
        claim_id = int(row["claim_id"])
        rows.append(_item(
            f"ambiguous-{claim_id}",
            "Without additional measurements, the exact numerical effect size "
            f"of the following general claim is known: {row['claim']}",
            None, "insufficient_information", "ambiguous",
            str(row["domain"]),
        ))

    if len(rows) != 360:
        raise RuntimeError(f"expected 360 benchmark items, got {len(rows)}")
    return rows


def benchmark_split(seed: int = 20260622) -> dict[str, list[str]]:
    """Stratified 50/50 validation/test split; test stays untouched."""
    generator = random.Random(seed)
    grouped: dict[tuple[str, int | None], list[str]] = {}
    for row in build_calibration_items():
        grouped.setdefault(
            (str(row["source"]), row["label"]), []
        ).append(str(row["item_id"]))
    validation: list[str] = []
    test: list[str] = []
    for ids in grouped.values():
        generator.shuffle(ids)
        midpoint = len(ids) // 2
        validation.extend(ids[:midpoint])
        test.extend(ids[midpoint:])
    if set(validation) & set(test):
        raise RuntimeError("benchmark validation/test overlap")
    return {"validation": sorted(validation), "test": sorted(test)}


def select_items(item_ids: list[str]) -> list[dict[str, Any]]:
    by_id = {str(row["item_id"]): row for row in build_calibration_items()}
    return [by_id[item_id] for item_id in item_ids]
