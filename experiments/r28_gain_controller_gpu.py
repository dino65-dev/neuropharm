"""Validation-trained bounded gain and abstaining-gate comparison for R2.8."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from experiments.r2_geometry_gpu import input_device, load_model, residual_layer
from experiments.r26_causal_and_dose_gpu import factual_metrics
from experiments.r26_design import MODEL_NAME
from experiments.r27_benchmark import benchmark_split, select_items
from experiments.r27_nuisance_optimization_gpu import build_partition_assets


ALPHA_LEVELS = (0.0, 0.03125, 0.0625, 0.125)


@contextmanager
def variable_additive_hook(
    model,
    layer: int,
    direction: torch.Tensor,
    prompt_lengths: list[int],
    answer_lengths: list[int],
    sequence_scales: list[float],
):
    def hook(module, args):
        residual = args[0]
        changed = residual.clone()
        vector = direction.to(residual.device, residual.dtype)
        for index, (prompt_length, answer_length, scale) in enumerate(
            zip(prompt_lengths, answer_lengths, sequence_scales)
        ):
            for response_index in range(answer_length):
                position = prompt_length - 1 + response_index
                changed[index, position, :] += vector * float(scale)
        return (changed, *args[1:])

    handle = residual_layer(model, layer).register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def evaluate_variable_scales(
    model,
    tokenizer,
    asset,
    direction: torch.Tensor,
    pair_scales: list[float],
    layer: int,
    batch_size: int,
) -> dict[str, list[float]]:
    examples = asset["examples"]
    if len(pair_scales) * 2 != len(examples):
        raise ValueError("one scale is required per response pair")
    tokenizer.pad_token = tokenizer.eos_token
    logprobs: list[float] = []
    sequence_kls: list[float] = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        max_length = max(row["token_ids"].shape[0] for row in batch)
        input_ids = torch.full(
            (len(batch), max_length), tokenizer.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        sequence_scales = []
        for local_index, row in enumerate(batch):
            length = row["token_ids"].shape[0]
            input_ids[local_index, :length] = row["token_ids"]
            attention_mask[local_index, :length] = 1
            sequence_scales.append(pair_scales[(start + local_index) // 2])
        prompt_lengths = [row["prompt_length"] for row in batch]
        answer_lengths = [row["answer_length"] for row in batch]
        with variable_additive_hook(
            model, layer, direction, prompt_lengths, answer_lengths,
            sequence_scales,
        ):
            with torch.inference_mode():
                full_logits = model(
                    input_ids=input_ids.to(input_device(model)),
                    attention_mask=attention_mask.to(input_device(model)),
                    use_cache=False,
                ).logits.float()
        for local_index, row in enumerate(batch):
            prompt_length = row["prompt_length"]
            answer_length = row["answer_length"]
            answer = row["token_ids"][
                prompt_length : prompt_length + answer_length
            ].to(full_logits.device)
            logits = full_logits[
                local_index,
                prompt_length - 1 : prompt_length - 1 + answer_length,
                :,
            ]
            log_p = torch.log_softmax(logits, dim=-1)
            logprobs.append(float(
                log_p.gather(-1, answer.unsqueeze(-1)).squeeze(-1).mean().item()
            ))
            baseline = asset["logits"][start + local_index].to(logits.device)
            log_q = torch.log_softmax(baseline, dim=-1)
            sequence_kls.append(float(
                torch.sum(log_p.exp() * (log_p - log_q), dim=-1).sum().item()
            ))
        del full_logits
    return {
        "scores": [
            logprobs[index] - logprobs[index + 1]
            for index in range(0, len(logprobs), 2)
        ],
        "kl": [
            (sequence_kls[index] + sequence_kls[index + 1]) / 2
            for index in range(0, len(sequence_kls), 2)
        ],
    }


def binary_entropy(score: float) -> float:
    probability = 1 / (1 + math.exp(-max(-60.0, min(60.0, score))))
    return -probability * math.log(max(probability, 1e-12)) - (
        1 - probability
    ) * math.log(max(1 - probability, 1e-12))


def feature_matrix(assets) -> tuple[torch.Tensor, list[str]]:
    factual = assets["factual"]
    count = len(factual["metadata"])
    prompt_lengths = [
        factual["examples"][2 * index]["prompt_length"]
        for index in range(count)
    ]
    rows = []
    for index in range(count):
        factual_score = factual["baseline"].scores[index]
        rows.append([
            factual_score,
            abs(factual_score),
            binary_entropy(factual_score),
            assets["definite"]["baseline"].scores[index],
            assets["numeric"]["baseline"].scores[index],
            assets["language"]["baseline"].scores[index],
            float(prompt_lengths[index]),
        ])
    return torch.tensor(rows, dtype=torch.float64), [
        "baseline_true_false_log_odds",
        "baseline_absolute_margin",
        "baseline_binary_entropy",
        "baseline_definite_log_odds",
        "baseline_numeric_confidence_log_odds",
        "baseline_language_log_odds",
        "prompt_token_length",
    ]


def fit_ridge(
    train_x: torch.Tensor, train_y: torch.Tensor, ridge: float = 1.0
) -> dict[str, torch.Tensor]:
    mean = train_x.mean(0)
    scale = train_x.std(0).clamp_min(1e-8)
    standardized = (train_x - mean) / scale
    design = torch.cat(
        (standardized, torch.ones(len(standardized), 1, dtype=train_x.dtype)),
        dim=1,
    )
    identity = torch.eye(design.shape[1], dtype=train_x.dtype)
    identity[-1, -1] = 0
    weights = torch.linalg.solve(
        design.T @ design + ridge * identity, design.T @ train_y
    )
    return {"mean": mean, "scale": scale, "weights": weights}


def predict(model: dict[str, torch.Tensor], values: torch.Tensor) -> torch.Tensor:
    standardized = (values - model["mean"]) / model["scale"]
    design = torch.cat(
        (standardized, torch.ones(len(values), 1, dtype=values.dtype)), dim=1
    )
    return design @ model["weights"]


def split_controller_validation(metadata) -> tuple[list[int], list[int]]:
    train = []
    tune = []
    for index, row in enumerate(metadata):
        checksum = sum(ord(character) for character in str(row["item_id"]))
        (tune if checksum % 3 == 0 else train).append(index)
    return train, tune


def policy_metrics(
    assets,
    endpoint_results: dict[str, dict[str, list[float]]],
    indices: list[int] | None = None,
) -> dict[str, Any]:
    if indices is None:
        indices = list(range(len(assets["factual"]["metadata"])))
    labels = [int(assets["factual"]["metadata"][index]["label"]) for index in indices]
    baseline_scores = [assets["factual"]["baseline"].scores[index] for index in indices]
    factual_scores = [endpoint_results["factual"]["scores"][index] for index in indices]
    baseline_metrics = factual_metrics(baseline_scores, labels)
    factual_current = factual_metrics(factual_scores, labels)
    correctness = [
        bool(assets["factual"]["metadata"][index]["baseline_correct"])
        for index in indices
    ]

    def deltas(name: str) -> list[float]:
        return [
            endpoint_results[name]["scores"][index]
            - assets[name]["baseline"].scores[index]
            for index in indices
        ]

    language = deltas("language")
    definite = deltas("definite")
    numeric = deltas("numeric")
    factual_margin = [
        (current - baseline) if label == 1 else -(current - baseline)
        for current, baseline, label in zip(
            factual_scores, baseline_scores, labels
        )
    ]
    wrong_numeric = [
        value for value, correct in zip(numeric, correctness) if not correct
    ]
    all_kl = [
        endpoint_results[name]["kl"][index]
        for name in ("language", "definite", "numeric", "factual")
        for index in indices
    ]
    return {
        "items": len(indices),
        "language_delta": statistics.fmean(language),
        "mean_absolute_language_delta": statistics.fmean(map(abs, language)),
        "definite_answer_delta": statistics.fmean(definite),
        "numeric_confidence_delta": statistics.fmean(numeric),
        "numeric_confidence_delta_baseline_incorrect": (
            statistics.fmean(wrong_numeric) if wrong_numeric else None
        ),
        "answer_margin_delta": statistics.fmean(factual_margin),
        "accuracy": factual_current["accuracy"],
        "accuracy_delta": (
            factual_current["accuracy"] - baseline_metrics["accuracy"]
        ),
        "brier_delta": factual_current["brier"] - baseline_metrics["brier"],
        "nll_delta": (
            factual_current["negative_log_likelihood"]
            - baseline_metrics["negative_log_likelihood"]
        ),
        "mean_kl": statistics.fmean(all_kl),
    }


def evaluate_policy(model, tokenizer, assets, direction, scales, layer, batch_size):
    return {
        name: evaluate_variable_scales(
            model, tokenizer, assets[name], direction, scales,
            layer, batch_size,
        )
        for name in ("language", "definite", "numeric", "factual")
    }


def is_safe_candidate(metrics: dict[str, Any]) -> bool:
    wrong = metrics["numeric_confidence_delta_baseline_incorrect"]
    return (
        metrics["brier_delta"] <= 0.0005
        and metrics["answer_margin_delta"] >= -0.001
        and (wrong is None or wrong <= 0.001)
    )


def utility(metrics: dict[str, Any]) -> float:
    # Used only after hard safety constraints pass.
    return (
        metrics["definite_answer_delta"]
        - metrics["mean_absolute_language_delta"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("basis", type=Path)
    parser.add_argument("slope_artifact", type=Path)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu-memory", default="14500MiB")
    parser.add_argument("--cpu-memory", default="12GiB")
    parser.add_argument(
        "--offload-folder", type=Path,
        default=Path("artifacts/_cache/r28_controller_offload"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/pkpd/r28_gain_controller_t4.json"),
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
    direction = basis["mean_receptor"].to(device, torch.float32)
    direction /= direction.norm().clamp_min(1e-12)
    slope_artifact = json.loads(
        args.slope_artifact.read_text(encoding="utf-8")
    )
    beta_by_id = {
        row["item_id"]: float(row["beta"])
        for row in slope_artifact["aggregated_across_epsilon"][
            "definite"
        ]["per_item"]
    }

    split = benchmark_split(args.seed)
    validation_assets = build_partition_assets(
        model, tokenizer, select_items(split["validation"]),
        args.layer, args.batch_size,
    )
    validation_x, feature_names = feature_matrix(validation_assets)
    validation_y = torch.tensor([
        beta_by_id[row["item_id"]]
        for row in validation_assets["factual"]["metadata"]
    ], dtype=torch.float64)
    controller_train, controller_tune = split_controller_validation(
        validation_assets["factual"]["metadata"]
    )
    ridge = fit_ridge(
        validation_x[controller_train], validation_y[controller_train]
    )
    validation_prediction = predict(ridge, validation_x)
    prediction_mean = validation_prediction[controller_train].mean()
    prediction_std = validation_prediction[controller_train].std().clamp_min(1e-8)

    fixed_results = {}
    fixed_metrics = {}
    for alpha in ALPHA_LEVELS:
        scales = [alpha] * len(validation_assets["factual"]["metadata"])
        results = evaluate_policy(
            model, tokenizer, validation_assets, direction, scales,
            args.layer, args.batch_size,
        )
        fixed_results[alpha] = results
        fixed_metrics[str(alpha)] = policy_metrics(
            validation_assets, results, controller_tune
        )
        print(f"validation fixed alpha {alpha:g}")
    safe_fixed = [
        alpha for alpha in ALPHA_LEVELS
        if is_safe_candidate(fixed_metrics[str(alpha)])
    ]
    best_fixed = max(
        safe_fixed or [0.0],
        key=lambda alpha: utility(fixed_metrics[str(alpha)]),
    )

    thresholds = [
        float(torch.quantile(
            validation_prediction[controller_train],
            torch.tensor(value, dtype=torch.float64),
        ).item())
        for value in (0.2, 0.4, 0.6, 0.8)
    ]
    gate_candidates = {}
    continuous_candidates = {}
    for alpha_max in (0.0625, 0.125):
        for threshold in thresholds:
            name = f"gate_a{alpha_max:g}_t{threshold:.6g}"
            scales = [
                alpha_max if value > threshold else 0.0
                for value in validation_prediction.tolist()
            ]
            results = evaluate_policy(
                model, tokenizer, validation_assets, direction, scales,
                args.layer, args.batch_size,
            )
            gate_candidates[name] = {
                "alpha_max": alpha_max,
                "threshold": threshold,
                "metrics": policy_metrics(
                    validation_assets, results, controller_tune
                ),
            }
        for temperature in (0.5, 1.0, 2.0):
            name = f"continuous_a{alpha_max:g}_temp{temperature:g}"
            z = (
                (validation_prediction - prediction_mean)
                / (prediction_std * temperature)
            )
            scales = (alpha_max * torch.sigmoid(z)).tolist()
            results = evaluate_policy(
                model, tokenizer, validation_assets, direction, scales,
                args.layer, args.batch_size,
            )
            continuous_candidates[name] = {
                "alpha_max": alpha_max,
                "temperature": temperature,
                "metrics": policy_metrics(
                    validation_assets, results, controller_tune
                ),
            }

    def choose(candidates):
        safe = [
            name for name, row in candidates.items()
            if is_safe_candidate(row["metrics"])
        ]
        return max(
            safe,
            key=lambda name: utility(candidates[name]["metrics"]),
            default=None,
        )

    best_gate_name = choose(gate_candidates)
    best_continuous_name = choose(continuous_candidates)
    print(f"selected fixed={best_fixed} gate={best_gate_name} continuous={best_continuous_name}")

    selected_validation_policies: dict[str, dict[str, Any]] = {}
    if best_gate_name is not None:
        gate = gate_candidates[best_gate_name]
        scales = [
            gate["alpha_max"] if value > gate["threshold"] else 0.0
            for value in validation_prediction.tolist()
        ]
        results = evaluate_policy(
            model, tokenizer, validation_assets, direction, scales,
            args.layer, args.batch_size,
        )
        selected_validation_policies["gated"] = {
            "scales": scales,
            "metrics": policy_metrics(validation_assets, results),
        }
    if best_continuous_name is not None:
        candidate = continuous_candidates[best_continuous_name]
        z = (
            (validation_prediction - prediction_mean)
            / (prediction_std * candidate["temperature"])
        )
        scales = (
            candidate["alpha_max"] * torch.sigmoid(z)
        ).tolist()
        results = evaluate_policy(
            model, tokenizer, validation_assets, direction, scales,
            args.layer, args.batch_size,
        )
        selected_validation_policies["continuous"] = {
            "scales": scales,
            "metrics": policy_metrics(validation_assets, results),
        }

    # Match each adaptive policy to a fixed intervention by measured validation
    # KL, not by nominal or mean dose.
    matched_fixed = {}
    for name, policy in selected_validation_policies.items():
        target_kl = policy["metrics"]["mean_kl"]
        lower = 0.0
        upper = 0.125
        best_alpha = 0.0
        best_metrics = policy_metrics(
            validation_assets, fixed_results[0.0]
        )
        for _ in range(7):
            midpoint = (lower + upper) / 2
            results = evaluate_policy(
                model, tokenizer, validation_assets, direction,
                [midpoint] * len(validation_x), args.layer, args.batch_size,
            )
            metrics = policy_metrics(validation_assets, results)
            if abs(metrics["mean_kl"] - target_kl) < abs(
                best_metrics["mean_kl"] - target_kl
            ):
                best_alpha = midpoint
                best_metrics = metrics
            if metrics["mean_kl"] < target_kl:
                lower = midpoint
            else:
                upper = midpoint
        matched_fixed[name] = {
            "alpha": best_alpha,
            "validation_metrics": best_metrics,
            "target_policy_kl": target_kl,
            "absolute_kl_error": abs(
                best_metrics["mean_kl"] - target_kl
            ),
        }

    # Test is loaded after all model and policy hyperparameters are fixed.
    test_assets = build_partition_assets(
        model, tokenizer, select_items(split["test"]),
        args.layer, args.batch_size,
    )
    test_x, _ = feature_matrix(test_assets)
    test_prediction = predict(ridge, test_x)
    policies: dict[str, list[float]] = {
        "zero": [0.0] * len(test_x),
        "best_fixed": [best_fixed] * len(test_x),
    }
    if best_gate_name is not None:
        gate = gate_candidates[best_gate_name]
        policies["gated"] = [
            gate["alpha_max"] if value > gate["threshold"] else 0.0
            for value in test_prediction.tolist()
        ]
    if best_continuous_name is not None:
        candidate = continuous_candidates[best_continuous_name]
        z = (
            (test_prediction - prediction_mean)
            / (prediction_std * candidate["temperature"])
        )
        policies["continuous"] = (
            candidate["alpha_max"] * torch.sigmoid(z)
        ).tolist()
    for name, row in matched_fixed.items():
        policies[f"kl_matched_fixed_for_{name}"] = [
            row["alpha"]
        ] * len(test_x)

    # Oracle is an explicit upper bound and may use correctness retrospectively.
    test_grid = {}
    for alpha in ALPHA_LEVELS:
        test_grid[alpha] = evaluate_policy(
            model, tokenizer, test_assets, direction,
            [alpha] * len(test_x), args.layer, args.batch_size,
        )
    oracle_scales = []
    labels = [
        int(row["label"]) for row in test_assets["factual"]["metadata"]
    ]
    correctness = [
        bool(row["baseline_correct"])
        for row in test_assets["factual"]["metadata"]
    ]
    for index in range(len(test_x)):
        candidates = []
        for alpha in ALPHA_LEVELS:
            result = test_grid[alpha]
            language_delta = (
                result["language"]["scores"][index]
                - test_assets["language"]["baseline"].scores[index]
            )
            definite_delta = (
                result["definite"]["scores"][index]
                - test_assets["definite"]["baseline"].scores[index]
            )
            numeric_delta = (
                result["numeric"]["scores"][index]
                - test_assets["numeric"]["baseline"].scores[index]
            )
            factual_delta = (
                result["factual"]["scores"][index]
                - test_assets["factual"]["baseline"].scores[index]
            ) * (1 if labels[index] == 1 else -1)
            safe = factual_delta >= -0.01 and (
                correctness[index] or numeric_delta <= 0
            )
            if safe:
                candidates.append((
                    definite_delta - abs(language_delta), alpha
                ))
        oracle_scales.append(max(candidates, default=(0.0, 0.0))[1])
    policies["oracle_upper_bound"] = oracle_scales

    test_metrics = {}
    scale_summaries = {}
    for name, scales in policies.items():
        results = evaluate_policy(
            model, tokenizer, test_assets, direction, scales,
            args.layer, args.batch_size,
        )
        test_metrics[name] = policy_metrics(test_assets, results)
        scale_summaries[name] = {
            "mean": statistics.fmean(scales),
            "zero_fraction": statistics.fmean(value == 0 for value in scales),
            "maximum": max(scales),
        }
        print(f"test policy {name}")

    prediction_correlation = float(torch.corrcoef(torch.stack((
        validation_prediction[controller_tune],
        validation_y[controller_tune],
    )))[0, 1].item())
    payload = {
        "experiment": "R2.8_bounded_gain_and_abstaining_gate",
        "model": MODEL_NAME,
        "layer": args.layer,
        "site": "block_input_residual_pre",
        "intervention": "direct additive alpha*r at scored response positions",
        "tolerance_or_homeostasis_enabled": False,
        "features": feature_names,
        "controller_training": {
            "validation_items": len(validation_x),
            "ridge_train_items": len(controller_train),
            "policy_tune_items": len(controller_tune),
            "test_items": len(test_x),
            "test_loaded_after_selection": True,
            "tune_prediction_correlation_with_definite_slope": prediction_correlation,
        },
        "selection_constraints": {
            "brier_delta_maximum": 0.0005,
            "answer_margin_delta_minimum": -0.001,
            "baseline_wrong_numeric_delta_maximum": 0.001,
            "utility_after_constraints": (
                "mean definite-answer delta minus mean absolute language delta"
            ),
        },
        "validation_fixed_metrics": fixed_metrics,
        "selected": {
            "best_fixed_alpha": best_fixed,
            "gate": (
                {**gate_candidates[best_gate_name], "name": best_gate_name}
                if best_gate_name else None
            ),
            "continuous": (
                {
                    **continuous_candidates[best_continuous_name],
                    "name": best_continuous_name,
                }
                if best_continuous_name else None
            ),
            "full_validation_policy_metrics": {
                name: row["metrics"]
                for name, row in selected_validation_policies.items()
            },
            "kl_matched_fixed_comparators": matched_fixed,
        },
        "untouched_test_metrics": test_metrics,
        "test_scale_summaries": scale_summaries,
        "oracle_note": (
            "The oracle uses ground-truth correctness retrospectively and is "
            "only an unattainable upper bound."
        ),
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
