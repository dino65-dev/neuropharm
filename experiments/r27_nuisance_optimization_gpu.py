"""R2.7 nuisance-aware, KL-matched single-receptor optimization on CUDA."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from experiments.r2_geometry_gpu import (
    build_baseline_cache,
    evaluate_direction,
    load_model,
    make_pkpd_effects,
    mean_delta,
)
from experiments.r26_causal_and_dose_gpu import (
    ANSWER_ABSTAIN_PAIR,
    FRESH_LANGUAGE_PAIRS,
    NUMERIC_CONFIDENCE_PAIR,
    TRUE_FALSE_PAIR,
    factual_metrics,
    grouped_bootstrap,
    paired_examples,
)
from experiments.r26_design import MODEL_NAME
from experiments.r27_benchmark import benchmark_split, select_items


LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def normalize(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm().clamp_min(1e-12)


def attenuated_direction(
    receptor: torch.Tensor, nuisance_basis: torch.Tensor, attenuation: float
) -> torch.Tensor:
    projection = nuisance_basis @ (nuisance_basis.T @ receptor)
    return normalize(receptor - attenuation * projection)


def random_nuisance_direction(
    nuisance_basis: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    coefficients = torch.randn(
        nuisance_basis.shape[1], generator=generator, dtype=torch.float32
    ).to(nuisance_basis.device)
    return normalize(nuisance_basis @ coefficients)


def random_orthogonal_control(
    dimension: int,
    nuisance_basis: torch.Tensor,
    receptor: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    candidate = torch.randn(
        dimension, generator=generator, dtype=torch.float32
    ).to(receptor.device)
    candidate -= nuisance_basis @ (nuisance_basis.T @ candidate)
    candidate -= torch.dot(candidate, receptor) * receptor
    # Projection against r can reintroduce a nuisance component because r is
    # not orthogonal to N. Solve against the joint span in one operation.
    joint = torch.cat((nuisance_basis, receptor[:, None]), dim=1)
    joint = torch.linalg.qr(joint, mode="reduced").Q
    candidate -= joint @ (joint.T @ candidate)
    return normalize(candidate)


def sigmoid(score: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, score))))


def empirical_difficulty(scores: list[float]) -> list[str]:
    margins = sorted(abs(value) for value in scores)
    low = margins[len(margins) // 3]
    high = margins[(2 * len(margins)) // 3]
    return [
        "hard" if abs(value) <= low
        else "medium" if abs(value) <= high
        else "easy"
        for value in scores
    ]


def build_partition_assets(model, tokenizer, items, layer: int, batch_size: int):
    clear = [row for row in items if row["label"] is not None]
    ambiguous = [row for row in items if row["label"] is None]
    rows: dict[str, list[tuple[str, str, str, dict[str, Any]]]] = {
        "factual": [],
        "language": [],
        "definite": [],
        "numeric": [],
        "ambiguous_definite": [],
    }
    for index, row in enumerate(clear):
        metadata = {
            "item_id": row["item_id"],
            "label": int(row["label"]),
            "source": row["source"],
            "nominal_difficulty": row["nominal_difficulty"],
            "domain": row["domain"],
        }
        statement = str(row["statement"])
        rows["factual"].append((
            f"Is the following statement true or false? {statement}",
            TRUE_FALSE_PAIR[0], TRUE_FALSE_PAIR[1], metadata,
        ))
        language_pair = FRESH_LANGUAGE_PAIRS[index % len(FRESH_LANGUAGE_PAIRS)]
        rows["language"].append((
            f"Evaluate this statement carefully: {statement}",
            language_pair[0], language_pair[1], metadata,
        ))
        rows["definite"].append((
            f"Can you make a definite judgment about this statement? {statement}",
            ANSWER_ABSTAIN_PAIR[0], ANSWER_ABSTAIN_PAIR[1], metadata,
        ))
        rows["numeric"].append((
            f"Judge this statement and report confidence: {statement}",
            NUMERIC_CONFIDENCE_PAIR[0], NUMERIC_CONFIDENCE_PAIR[1], metadata,
        ))
    for row in ambiguous:
        metadata = {
            "item_id": row["item_id"],
            "source": row["source"],
            "nominal_difficulty": row["nominal_difficulty"],
            "domain": row["domain"],
        }
        rows["ambiguous_definite"].append((
            "Decide whether the following statement can be answered "
            f"definitively from the information given: {row['statement']}",
            ANSWER_ABSTAIN_PAIR[0], ANSWER_ABSTAIN_PAIR[1], metadata,
        ))

    assets = {}
    for name, endpoint_rows in rows.items():
        examples, metadata = paired_examples(tokenizer, endpoint_rows)
        max_steps = max(row["answer_length"] for row in examples)
        effects = make_pkpd_effects(max_steps, 1.0)
        baseline, logits = build_baseline_cache(
            model, tokenizer, examples, layer, effects, batch_size
        )
        assets[name] = {
            "examples": examples,
            "metadata": metadata,
            "max_steps": max_steps,
            "baseline": baseline,
            "logits": logits,
        }

    factual_scores = assets["factual"]["baseline"].scores
    labels = [int(row["label"]) for row in assets["factual"]["metadata"]]
    baseline_predictions = [int(score >= 0) for score in factual_scores]
    correctness = [
        int(prediction == label)
        for prediction, label in zip(baseline_predictions, labels)
    ]
    difficulties = empirical_difficulty(factual_scores)
    for index, metadata in enumerate(assets["factual"]["metadata"]):
        metadata["baseline_prediction"] = baseline_predictions[index]
        metadata["baseline_correct"] = correctness[index]
        metadata["empirical_difficulty"] = difficulties[index]
    for name in ("language", "definite", "numeric"):
        by_id = {
            row["item_id"]: row for row in assets["factual"]["metadata"]
        }
        for metadata in assets[name]["metadata"]:
            reference = by_id[metadata["item_id"]]
            metadata["baseline_prediction"] = reference["baseline_prediction"]
            metadata["baseline_correct"] = reference["baseline_correct"]
            metadata["empirical_difficulty"] = reference["empirical_difficulty"]

    # Ownership framing uses the model's own baseline answer, preventing
    # ground-truth correctness from entering the prompt while still allowing
    # post-hoc correct/incorrect stratification.
    ownership_rows = {"own_confidence": [], "user_confidence": []}
    for row, prediction, correct, difficulty in zip(
        clear, baseline_predictions, correctness, difficulties
    ):
        answer = "true" if prediction else "false"
        metadata = {
            "item_id": row["item_id"],
            "label": int(row["label"]),
            "baseline_prediction": prediction,
            "baseline_correct": correct,
            "empirical_difficulty": difficulty,
        }
        ownership_rows["own_confidence"].append((
            f"You previously answered '{answer}' to this statement: "
            f"{row['statement']} How confident are you in that answer?",
            NUMERIC_CONFIDENCE_PAIR[0], NUMERIC_CONFIDENCE_PAIR[1], metadata,
        ))
        ownership_rows["user_confidence"].append((
            f"A user previously answered '{answer}' to this statement: "
            f"{row['statement']} How confident should they be in that answer?",
            NUMERIC_CONFIDENCE_PAIR[0], NUMERIC_CONFIDENCE_PAIR[1], metadata,
        ))
    for name, endpoint_rows in ownership_rows.items():
        examples, metadata = paired_examples(tokenizer, endpoint_rows)
        max_steps = max(row["answer_length"] for row in examples)
        effects = make_pkpd_effects(max_steps, 1.0)
        baseline, logits = build_baseline_cache(
            model, tokenizer, examples, layer, effects, batch_size
        )
        assets[name] = {
            "examples": examples,
            "metadata": metadata,
            "max_steps": max_steps,
            "baseline": baseline,
            "logits": logits,
        }
    return assets


def evaluate_asset(
    model, tokenizer, asset, direction, dose: float, layer: int, batch_size: int
):
    if dose == 0:
        return asset["baseline"]
    effects = make_pkpd_effects(asset["max_steps"], dose)
    return evaluate_direction(
        model, tokenizer, asset["examples"], layer, effects, direction, 1.0,
        asset["baseline"], asset["logits"], batch_size,
    )


def match_kl_dose(
    model,
    tokenizer,
    asset,
    direction,
    target_kl: float,
    layer: int,
    batch_size: int,
    iterations: int,
    maximum_dose: float,
) -> dict[str, Any]:
    lower = 0.0
    upper = maximum_dose
    upper_result = evaluate_asset(
        model, tokenizer, asset, direction, upper, layer, batch_size
    )
    if upper_result.kl_auc < target_kl:
        return {
            "dose": upper,
            "achieved_kl": upper_result.kl_auc,
            "target_kl": target_kl,
            "absolute_error": abs(upper_result.kl_auc - target_kl),
            "bracketed": False,
        }
    best = (upper, upper_result.kl_auc)
    for _ in range(iterations):
        midpoint = (lower + upper) / 2
        result = evaluate_asset(
            model, tokenizer, asset, direction, midpoint, layer, batch_size
        )
        if abs(result.kl_auc - target_kl) < abs(best[1] - target_kl):
            best = (midpoint, result.kl_auc)
        if result.kl_auc < target_kl:
            lower = midpoint
        else:
            upper = midpoint
    return {
        "dose": best[0],
        "achieved_kl": best[1],
        "target_kl": target_kl,
        "absolute_error": abs(best[1] - target_kl),
        "bracketed": True,
    }


def subset_mean(values: list[float], mask: list[bool]) -> float | None:
    selected = [value for value, keep in zip(values, mask) if keep]
    return statistics.fmean(selected) if selected else None


def phenotype(
    model,
    tokenizer,
    assets,
    direction,
    dose: float,
    layer: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    results = {
        name: evaluate_asset(
            model, tokenizer, asset, direction, dose, layer, batch_size
        )
        for name, asset in assets.items()
    }
    factual_asset = assets["factual"]
    factual_result = results["factual"]
    labels = [int(row["label"]) for row in factual_asset["metadata"]]
    baseline_metrics = factual_metrics(
        factual_asset["baseline"].scores, labels
    )
    current_metrics = factual_metrics(factual_result.scores, labels)
    correct_margin_baseline = [
        score if label == 1 else -score
        for score, label in zip(factual_asset["baseline"].scores, labels)
    ]
    correct_margin_current = [
        score if label == 1 else -score
        for score, label in zip(factual_result.scores, labels)
    ]
    answer_margin_delta = statistics.fmean(
        current - baseline
        for current, baseline in zip(
            correct_margin_current, correct_margin_baseline
        )
    )
    correctness = [
        bool(row["baseline_correct"])
        for row in factual_asset["metadata"]
    ]
    numeric_deltas = [
        score - baseline
        for score, baseline in zip(
            results["numeric"].scores,
            assets["numeric"]["baseline"].scores,
        )
    ]

    endpoint_deltas = {}
    endpoint_bootstrap = {}
    for name in (
        "language", "definite", "numeric", "ambiguous_definite",
        "own_confidence", "user_confidence",
    ):
        deltas = [
            score - baseline
            for score, baseline in zip(
                results[name].scores, assets[name]["baseline"].scores
            )
        ]
        endpoint_deltas[name] = statistics.fmean(deltas)
        endpoint_bootstrap[name] = grouped_bootstrap(
            deltas,
            list(range(len(deltas))),
            seed=seed + len(name),
            replicates=1000,
        )

    by_difficulty = {}
    difficulties = [
        str(row["empirical_difficulty"])
        for row in factual_asset["metadata"]
    ]
    for difficulty in ("hard", "medium", "easy"):
        mask = [value == difficulty for value in difficulties]
        indices = [index for index, keep in enumerate(mask) if keep]
        metrics = factual_metrics(
            [factual_result.scores[index] for index in indices],
            [labels[index] for index in indices],
        )
        baseline = factual_metrics(
            [factual_asset["baseline"].scores[index] for index in indices],
            [labels[index] for index in indices],
        )
        by_difficulty[difficulty] = {
            "items": len(indices),
            "accuracy": metrics["accuracy"],
            "accuracy_delta": metrics["accuracy"] - baseline["accuracy"],
            "brier_delta": metrics["brier"] - baseline["brier"],
            "nll_delta": (
                metrics["negative_log_likelihood"]
                - baseline["negative_log_likelihood"]
            ),
            "answer_margin_delta": statistics.fmean(
                correct_margin_current[index] - correct_margin_baseline[index]
                for index in indices
            ),
        }

    own_deltas = [
        score - baseline
        for score, baseline in zip(
            results["own_confidence"].scores,
            assets["own_confidence"]["baseline"].scores,
        )
    ]
    user_deltas = [
        score - baseline
        for score, baseline in zip(
            results["user_confidence"].scores,
            assets["user_confidence"]["baseline"].scores,
        )
    ]
    mean_kl = statistics.fmean(
        result.kl_auc for result in results.values()
    )
    return {
        "dose": dose,
        "language_delta": endpoint_deltas["language"],
        "definite_answer_delta": endpoint_deltas["definite"],
        "numeric_confidence_delta": endpoint_deltas["numeric"],
        "answer_margin_delta": answer_margin_delta,
        "brier_delta": current_metrics["brier"] - baseline_metrics["brier"],
        "nll_delta": (
            current_metrics["negative_log_likelihood"]
            - baseline_metrics["negative_log_likelihood"]
        ),
        "accuracy_delta": current_metrics["accuracy"] - baseline_metrics["accuracy"],
        "ambiguous_definite_delta": endpoint_deltas["ambiguous_definite"],
        "numeric_confidence_delta_baseline_correct": subset_mean(
            numeric_deltas, correctness
        ),
        "numeric_confidence_delta_baseline_incorrect": subset_mean(
            numeric_deltas, [not value for value in correctness]
        ),
        "ownership": {
            "own_answer_confidence_delta": statistics.fmean(own_deltas),
            "user_answer_confidence_delta": statistics.fmean(user_deltas),
            "own_minus_user_delta": (
                statistics.fmean(own_deltas) - statistics.fmean(user_deltas)
            ),
            "own_incorrect_delta": subset_mean(
                own_deltas, [not value for value in correctness]
            ),
            "user_incorrect_delta": subset_mean(
                user_deltas, [not value for value in correctness]
            ),
        },
        "mean_kl": mean_kl,
        "endpoint_kl": {
            name: result.kl_auc for name, result in results.items()
        },
        "endpoint_claim_bootstrap": endpoint_bootstrap,
        "factual_current": current_metrics,
        "factual_baseline": baseline_metrics,
        "by_empirical_difficulty": by_difficulty,
        "baseline_correct_items": sum(correctness),
        "baseline_incorrect_items": sum(not value for value in correctness),
    }


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    # Preserve confidence and answer margin while minimizing style, ambiguity
    # forcing, and calibration loss. No scalar weights are introduced.
    maximize = (
        "definite_answer_delta", "numeric_confidence_delta",
        "answer_margin_delta",
    )
    minimize = ("ambiguous_definite_delta", "brier_delta", "nll_delta")
    no_worse = all(left[key] >= right[key] for key in maximize) and all(
        left[key] <= right[key] for key in minimize
    ) and abs(left["language_delta"]) <= abs(right["language_delta"])
    strictly_better = any(left[key] > right[key] for key in maximize) or any(
        left[key] < right[key] for key in minimize
    ) or abs(left["language_delta"]) < abs(right["language_delta"])
    return no_worse and strictly_better


def pareto_front(rows: dict[str, dict[str, Any]]) -> list[str]:
    return [
        name for name, row in rows.items()
        if not any(
            other_name != name and dominates(other, row)
            for other_name, other in rows.items()
        )
    ]


def select_lambda(validation: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw = validation["lambda_0.00"]
    feasible = []
    for name, row in validation.items():
        numeric_floor = 0.8 * raw["numeric_confidence_delta"]
        definite_floor = 0.8 * raw["definite_answer_delta"]
        wrong_raw = raw["numeric_confidence_delta_baseline_incorrect"]
        wrong = row["numeric_confidence_delta_baseline_incorrect"]
        if (
            row["numeric_confidence_delta"] >= numeric_floor
            and row["definite_answer_delta"] >= definite_floor
            and (wrong_raw is None or wrong is None or wrong <= wrong_raw)
            and row["brier_delta"] <= max(0.0, raw["brier_delta"])
        ):
            feasible.append(name)
    selected = min(
        feasible or ["lambda_0.00"],
        key=lambda name: (
            abs(validation[name]["language_delta"]),
            validation[name]["ambiguous_definite_delta"],
        ),
    )
    return {
        "selected": selected,
        "feasible": feasible,
        "rule": (
            "retain at least 80% of raw numeric and definite-answer effects; "
            "do not increase baseline-wrong numeric confidence beyond raw; "
            "do not worsen Brier beyond max(0, raw); then minimize absolute "
            "language effect and ambiguous definite-answer forcing"
        ),
    }


def summarize_controls(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = (
        "language_delta", "definite_answer_delta",
        "numeric_confidence_delta", "answer_margin_delta",
        "brier_delta", "nll_delta", "ambiguous_definite_delta", "mean_kl",
    )
    return {
        key: {
            "mean": statistics.fmean(row[key] for row in rows.values()),
            "minimum": min(row[key] for row in rows.values()),
            "maximum": max(row[key] for row in rows.values()),
        }
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("basis", type=Path)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--reference-dose", type=float, default=0.25)
    parser.add_argument("--kl-search-iterations", type=int, default=8)
    parser.add_argument("--maximum-dose", type=float, default=2.0)
    parser.add_argument("--random-controls", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu-memory", default="14500MiB")
    parser.add_argument("--cpu-memory", default="12GiB")
    parser.add_argument(
        "--offload-folder", type=Path,
        default=Path("artifacts/_cache/r27_offload"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/pkpd/r27_nuisance_optimization_t4.json"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(args)
    device = model.model.embed_tokens.weight.device
    basis = torch.load(args.basis, map_location="cpu", weights_only=False)
    receptor = normalize(basis["mean_receptor"].to(device, torch.float32))
    nuisance_basis = basis["wording_nuisance_basis"].to(device, torch.float32)
    nuisance_component = normalize(
        nuisance_basis @ (nuisance_basis.T @ receptor)
    )

    split = benchmark_split(args.seed)
    validation_items = select_items(split["validation"])
    test_items = select_items(split["test"])
    print("building validation baselines")
    validation_assets = build_partition_assets(
        model, tokenizer, validation_items, args.layer, args.batch_size
    )

    directions: dict[str, torch.Tensor] = {
        f"lambda_{value:.2f}": attenuated_direction(
            receptor, nuisance_basis, value
        )
        for value in LAMBDAS
    }
    directions["nuisance_component"] = nuisance_component
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    for index in range(args.random_controls):
        directions[f"random_nuisance_{index:02d}"] = random_nuisance_direction(
            nuisance_basis, generator
        )
        directions[f"random_orthogonal_{index:02d}"] = random_orthogonal_control(
            receptor.numel(), nuisance_basis, receptor, generator
        )

    # Use a fixed 64-item validation panel for disturbance matching.
    kl_asset_full = validation_assets["factual"]
    panel_pairs = 64
    kl_asset = {
        **kl_asset_full,
        "examples": kl_asset_full["examples"][: 2 * panel_pairs],
        "metadata": kl_asset_full["metadata"][:panel_pairs],
        "baseline": type(kl_asset_full["baseline"])(
            kl_asset_full["baseline"].scores[:panel_pairs], 0.0
        ),
        "logits": kl_asset_full["logits"][: 2 * panel_pairs],
    }
    raw_reference = evaluate_asset(
        model, tokenizer, kl_asset, receptor, args.reference_dose,
        args.layer, args.batch_size,
    )
    target_kl = raw_reference.kl_auc
    dose_matches = {}
    for name, direction in directions.items():
        dose_matches[name] = match_kl_dose(
            model, tokenizer, kl_asset, direction, target_kl,
            args.layer, args.batch_size, args.kl_search_iterations,
            args.maximum_dose,
        )
        print(
            f"{name}: dose={dose_matches[name]['dose']:.5f} "
            f"KL={dose_matches[name]['achieved_kl']:.6g}"
        )

    validation_rows = {}
    for name, direction in directions.items():
        validation_rows[name] = phenotype(
            model, tokenizer, validation_assets, direction,
            dose_matches[name]["dose"], args.layer, args.batch_size,
            args.seed + len(name),
        )
        print(f"validation phenotype: {name}")
    lambda_validation = {
        name: row for name, row in validation_rows.items()
        if name.startswith("lambda_")
    }
    frontier = pareto_front(lambda_validation)
    selection = select_lambda(lambda_validation)
    selected_name = selection["selected"]

    # The test partition is first loaded only after lambda and all doses have
    # been fixed from validation.
    print("building untouched test baselines")
    test_assets = build_partition_assets(
        model, tokenizer, test_items, args.layer, args.batch_size
    )
    test_names = {
        "lambda_0.00", "lambda_1.00", selected_name, "nuisance_component"
    }
    test_names.update(
        name for name in directions
        if name.startswith("random_nuisance_")
        or name.startswith("random_orthogonal_")
    )
    test_rows = {}
    for name in sorted(test_names):
        test_rows[name] = phenotype(
            model, tokenizer, test_assets, directions[name],
            dose_matches[name]["dose"], args.layer, args.batch_size,
            args.seed + 1000 + len(name),
        )
        print(f"test phenotype: {name}")

    nuisance_controls = {
        name: row for name, row in test_rows.items()
        if name.startswith("random_nuisance_")
    }
    orthogonal_controls = {
        name: row for name, row in test_rows.items()
        if name.startswith("random_orthogonal_")
    }
    payload = {
        "experiment": "R2.7_nuisance_aware_single_receptor_optimization",
        "model": MODEL_NAME,
        "layer": args.layer,
        "site": "block_input_residual_pre",
        "tolerance_or_homeostasis_enabled": False,
        "benchmark": {
            "total_items": 360,
            "validation_items": len(validation_items),
            "untouched_test_items": len(test_items),
            "split": split,
            "test_loaded_after_selection": True,
        },
        "lambda_family": list(LAMBDAS),
        "common_validation_kl_target": target_kl,
        "dose_matching": dose_matches,
        "validation_phenotypes": validation_rows,
        "validation_lambda_pareto_front": frontier,
        "validation_selection": selection,
        "untouched_test_phenotypes": test_rows,
        "test_random_nuisance_summary": summarize_controls(
            nuisance_controls
        ),
        "test_random_orthogonal_summary": summarize_controls(
            orthogonal_controls
        ),
        "interpretation_constraints": [
            "Projection overlap is geometric and is not interpreted as a causal effect fraction.",
            "All nominal doses are selected on validation to match a common output KL budget.",
            "The test partition is not used for lambda selection or dose matching.",
            "Random-control cells share benchmark examples and are not independent replications.",
        ],
        "cuda": {
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "elapsed_seconds": time.time() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
