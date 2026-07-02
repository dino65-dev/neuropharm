"""R2.8 local causal slopes and KL curvature for a fixed receptor."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from experiments.r2_geometry_gpu import (
    batched_intervention_hook,
    input_device,
    load_model,
)
from experiments.r26_design import MODEL_NAME
from experiments.r27_benchmark import benchmark_split, select_items
from experiments.r27_nuisance_optimization_gpu import build_partition_assets


EPSILONS = (0.03125, 0.0625, 0.125)


def run_pair_scores_and_kl(
    model,
    tokenizer,
    asset,
    layer: int,
    direction: torch.Tensor,
    epsilon: float,
    sign: float,
    batch_size: int,
) -> tuple[list[float], list[float]]:
    examples = asset["examples"]
    # R2.8 estimates the local derivative with respect to additive steering
    # amplitude. Do not pass epsilon through the nonlinear Hill PK/PD map.
    effects = [epsilon] * asset["max_steps"]
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
        for index, row in enumerate(batch):
            length = row["token_ids"].shape[0]
            input_ids[index, :length] = row["token_ids"]
            attention_mask[index, :length] = 1
        prompt_lengths = [row["prompt_length"] for row in batch]
        answer_lengths = [row["answer_length"] for row in batch]
        with batched_intervention_hook(
            model, layer, direction, effects, sign,
            prompt_lengths, answer_lengths,
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
    pair_scores = [
        logprobs[index] - logprobs[index + 1]
        for index in range(0, len(logprobs), 2)
    ]
    pair_kls = [
        (sequence_kls[index] + sequence_kls[index + 1]) / 2
        for index in range(0, len(sequence_kls), 2)
    ]
    return pair_scores, pair_kls


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 3:
        return None
    x = torch.tensor(left, dtype=torch.float64)
    y = torch.tensor(right, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.norm() * y.norm()
    if denominator <= 1e-30:
        return None
    return float(torch.dot(x, y).div(denominator).item())


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    n = len(values)
    mean = statistics.fmean(values)
    return {
        "mean": mean,
        "variance": statistics.variance(values) if n > 1 else 0.0,
        "standard_deviation": statistics.stdev(values) if n > 1 else 0.0,
        "coefficient_of_variation": (
            statistics.stdev(values) / abs(mean)
            if n > 1 and abs(mean) > 1e-12 else float("inf")
        ),
        "p05": ordered[max(0, math.ceil(0.05 * n) - 1)],
        "median": statistics.median(values),
        "p95": ordered[min(n - 1, math.ceil(0.95 * n) - 1)],
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "positive_fraction": statistics.fmean(value > 0 for value in values),
    }


def grouped_summary(
    values: list[float], groups: list[str]
) -> dict[str, dict[str, float]]:
    return {
        group: summarize([
            value for value, observed in zip(values, groups)
            if observed == group
        ])
        for group in sorted(set(groups))
    }


def binary_entropy_from_log_odds(score: float) -> float:
    probability = 1.0 / (
        1.0 + math.exp(-max(-60.0, min(60.0, score)))
    )
    return -(
        probability * math.log(max(probability, 1e-12))
        + (1 - probability) * math.log(max(1 - probability, 1e-12))
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("basis", type=Path)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu-memory", default="14500MiB")
    parser.add_argument("--cpu-memory", default="12GiB")
    parser.add_argument(
        "--offload-folder", type=Path,
        default=Path("artifacts/_cache/r28_slopes_offload"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/pkpd/r28_context_slopes_t4.json"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(args)
    device = model.model.embed_tokens.weight.device
    bundle = torch.load(args.basis, map_location="cpu", weights_only=False)
    receptor = bundle["mean_receptor"].to(device, torch.float32)
    receptor = receptor / receptor.norm().clamp_min(1e-12)

    validation_items = select_items(benchmark_split(args.seed)["validation"])
    assets = build_partition_assets(
        model, tokenizer, validation_items, args.layer, args.batch_size
    )
    factual_metadata = assets["factual"]["metadata"]
    item_ids = [str(row["item_id"]) for row in factual_metadata]
    labels = [int(row["label"]) for row in factual_metadata]
    domains = [str(row["domain"]) for row in factual_metadata]
    difficulties = [
        str(row["empirical_difficulty"]) for row in factual_metadata
    ]
    baseline_factual = assets["factual"]["baseline"].scores
    baseline_uncertainty = [
        binary_entropy_from_log_odds(score) for score in baseline_factual
    ]
    baseline_margin = [abs(score) for score in baseline_factual]
    baseline_correct = [
        bool(row["baseline_correct"]) for row in factual_metadata
    ]

    endpoint_names = ("language", "definite", "numeric", "factual")
    epsilon_rows: dict[str, Any] = {}
    per_endpoint_beta: dict[str, list[list[float]]] = {
        name: [] for name in endpoint_names
    }
    per_endpoint_kappa: dict[str, list[list[float]]] = {
        name: [] for name in endpoint_names
    }
    for epsilon in EPSILONS:
        epsilon_payload = {}
        for name in endpoint_names:
            positive_scores, positive_kl = run_pair_scores_and_kl(
                model, tokenizer, assets[name], args.layer, receptor,
                epsilon, 1.0, args.batch_size,
            )
            negative_scores, negative_kl = run_pair_scores_and_kl(
                model, tokenizer, assets[name], args.layer, receptor,
                epsilon, -1.0, args.batch_size,
            )
            beta = [
                (positive - negative) / (2 * epsilon)
                for positive, negative in zip(positive_scores, negative_scores)
            ]
            if name == "factual":
                beta = [
                    value if label == 1 else -value
                    for value, label in zip(beta, labels)
                ]
            kappa = [
                (positive + negative) / (2 * epsilon * epsilon)
                for positive, negative in zip(positive_kl, negative_kl)
            ]
            efficiency = [
                slope / math.sqrt(curvature + 1e-12)
                for slope, curvature in zip(beta, kappa)
            ]
            per_endpoint_beta[name].append(beta)
            per_endpoint_kappa[name].append(kappa)
            epsilon_payload[name] = {
                "beta": summarize(beta),
                "kappa": summarize(kappa),
                "efficiency": summarize(efficiency),
            }
        epsilon_rows[str(epsilon)] = epsilon_payload
        print(f"epsilon {epsilon:g}")

    aggregated = {}
    for name in endpoint_names:
        beta = [
            statistics.fmean(rows[index] for rows in per_endpoint_beta[name])
            for index in range(len(per_endpoint_beta[name][0]))
        ]
        kappa = [
            statistics.fmean(rows[index] for rows in per_endpoint_kappa[name])
            for index in range(len(per_endpoint_kappa[name][0]))
        ]
        efficiency = [
            slope / math.sqrt(curvature + 1e-12)
            for slope, curvature in zip(beta, kappa)
        ]
        aggregated[name] = {
            "beta": summarize(beta),
            "kappa": summarize(kappa),
            "efficiency": summarize(efficiency),
            "beta_by_domain": grouped_summary(beta, domains),
            "beta_by_empirical_difficulty": grouped_summary(
                beta, difficulties
            ),
            "beta_by_baseline_correctness": grouped_summary(
                beta, [
                    "correct" if value else "incorrect"
                    for value in baseline_correct
                ],
            ),
            "correlation_beta_with_baseline_entropy": pearson(
                beta, baseline_uncertainty
            ),
            "correlation_beta_with_absolute_margin": pearson(
                beta, baseline_margin
            ),
            "correlation_efficiency_with_baseline_entropy": pearson(
                efficiency, baseline_uncertainty
            ),
            "per_item": [
                {
                    "item_id": item_id,
                    "domain": domain,
                    "difficulty": difficulty,
                    "baseline_correct": correct,
                    "baseline_entropy": entropy,
                    "baseline_absolute_margin": margin,
                    "beta": slope,
                    "kappa": curvature,
                    "efficiency": eta,
                }
                for (
                    item_id, domain, difficulty, correct, entropy, margin,
                    slope, curvature, eta,
                ) in zip(
                    item_ids, domains, difficulties, baseline_correct,
                    baseline_uncertainty, baseline_margin, beta, kappa,
                    efficiency,
                )
            ],
        }

    target = aggregated["definite"]
    epsilon_positive = [
        epsilon_rows[str(value)]["definite"]["beta"]["positive_fraction"]
        for value in EPSILONS
    ]
    stable_sign = min(epsilon_positive) >= 0.8
    variable_magnitude = (
        target["beta"]["coefficient_of_variation"] >= 0.5
        or (
            target["beta"]["p95"] - target["beta"]["p05"]
            > abs(target["beta"]["mean"])
        )
    )
    frequent_reversal = min(epsilon_positive) < 0.7
    if stable_sign and variable_magnitude:
        outcome = "stable_sign_variable_magnitude"
        recommendation = (
            "A bounded nonnegative context-dependent gain is justified for "
            "validation testing with fixed receptor direction."
        )
    elif frequent_reversal:
        outcome = "frequent_sign_reversals"
        recommendation = (
            "Do not steer every example. Test an abstaining gate before any "
            "continuous gain controller."
        )
    else:
        outcome = "weak_or_inconclusive_heterogeneity"
        recommendation = (
            "Retain fixed dosing until a larger validation study resolves "
            "slope sign and magnitude heterogeneity."
        )
    payload = {
        "experiment": "R2.8_fixed_receptor_local_slopes",
        "model": MODEL_NAME,
        "layer": args.layer,
        "site": "block_input_residual_pre",
        "epsilons": list(EPSILONS),
        "intervention_parameterization": (
            "direct additive block-input residual perturbation +/- epsilon*r "
            "at every scored response position; no Hill occupancy transform"
        ),
        "validation_items": len(validation_items),
        "clear_labeled_items": len(item_ids),
        "epsilon_results": epsilon_rows,
        "aggregated_across_epsilon": aggregated,
        "gain_gate": {
            "target_endpoint": "definite_answer_log_odds",
            "minimum_positive_fraction_across_epsilon": min(epsilon_positive),
            "stable_sign_threshold": 0.8,
            "frequent_reversal_threshold": 0.7,
            "variable_magnitude": variable_magnitude,
            "outcome": outcome,
            "recommendation": recommendation,
        },
        "constraints": [
            "Only validation items are used; the R2.7 test partition remains untouched.",
            "Ground-truth correctness is used only for retrospective stratification, never as a controller feature.",
            "Slopes are derivatives with respect to direct additive residual amplitude, not nominal PK/PD concentration.",
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
