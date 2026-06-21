"""R2.6 unseen-template causal transfer and mean-receptor dose response."""
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
    ScoreResult,
    antisymmetric_effect,
    build_baseline_cache,
    evaluate_direction,
    load_model,
    make_pkpd_effects,
    mean_delta,
    response_ids,
)
from experiments.r26_design import (
    MODEL_NAME,
    QUESTION_TEMPLATES,
    RESPONSE_FRAMES,
)


DOSES = (0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0)
FRESH_LANGUAGE_PAIRS = (
    ("The conclusion is beyond reasonable doubt.", "The conclusion remains open to substantial doubt."),
    ("I would state this conclusion without reservation.", "I would state this conclusion with substantial reservation."),
    ("The conclusion can be treated as firmly established.", "The conclusion should be treated as provisional."),
    ("There is little reason to question this conclusion.", "There is considerable reason to question this conclusion."),
)
ANSWER_ABSTAIN_PAIR = (
    "I can give a definite answer to this question.",
    "I should abstain because I cannot give a definite answer.",
)
NUMERIC_CONFIDENCE_PAIR = ("Confidence: 90%.", "Confidence: 50%.")
TRUE_FALSE_PAIR = ("True.", "False.")


def chat_prompt_ids(tokenizer, prompt: str) -> torch.Tensor:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(encoded, torch.Tensor):
        return encoded
    if hasattr(encoded, "input_ids"):
        return encoded.input_ids
    return encoded["input_ids"]


def paired_examples(
    tokenizer, rows: list[tuple[str, str, str, dict[str, Any]]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build two-response examples and aligned pair metadata."""
    examples = []
    metadata = []
    for pair_index, (prompt, first, second, row_metadata) in enumerate(rows):
        prompt_tokens = chat_prompt_ids(tokenizer, prompt)
        for sign_index, response in enumerate((first, second)):
            answer_tokens = response_ids(tokenizer, response)
            token_ids = torch.cat((prompt_tokens, answer_tokens), dim=1)
            examples.append({
                "row_index": pair_index,
                "pair_index": pair_index,
                "sign_index": sign_index,
                "token_ids": token_ids.squeeze(0),
                "prompt_length": int(prompt_tokens.shape[1]),
                "answer_length": int(answer_tokens.shape[1]),
            })
        metadata.append(row_metadata)
    return examples, metadata


def normalize(vector: torch.Tensor) -> torch.Tensor:
    return vector / vector.norm().clamp_min(1e-12)


def evaluate_signed(
    model,
    tokenizer,
    examples,
    baseline,
    baseline_logits,
    direction,
    effects,
    layer,
    batch_size,
    pair_metadata,
    seed,
) -> dict[str, Any]:
    positive = evaluate_direction(
        model, tokenizer, examples, layer, effects, direction, 1.0,
        baseline, baseline_logits, batch_size,
    )
    negative = evaluate_direction(
        model, tokenizer, examples, layer, effects, -direction, 1.0,
        baseline, baseline_logits, batch_size,
    )
    pair_effects = [
        ((positive.scores[index] - baseline.scores[index])
         - (negative.scores[index] - baseline.scores[index])) / 2
        for index in range(len(baseline.scores))
    ]
    claim_ids = [int(row["claim_id"]) for row in pair_metadata]
    return {
        "positive_delta": mean_delta(positive, baseline),
        "negative_delta": mean_delta(negative, baseline),
        "antisymmetric_effect": antisymmetric_effect(
            positive, negative, baseline
        ),
        "mean_kl": (positive.kl_auc + negative.kl_auc) / 2,
        "claim_block_bootstrap": grouped_bootstrap(
            pair_effects, claim_ids, seed=seed
        ),
    }


def grouped_bootstrap(
    values: list[float],
    claim_ids: list[int],
    seed: int,
    replicates: int = 2000,
) -> dict[str, float]:
    by_claim: dict[int, list[float]] = {}
    for value, claim_id in zip(values, claim_ids):
        by_claim.setdefault(claim_id, []).append(float(value))
    claim_means = {
        claim_id: statistics.fmean(rows)
        for claim_id, rows in by_claim.items()
    }
    ids = sorted(claim_means)
    generator = random.Random(seed)
    samples = []
    for _ in range(replicates):
        sampled = [ids[generator.randrange(len(ids))] for _ in ids]
        samples.append(statistics.fmean(claim_means[index] for index in sampled))
    ordered = sorted(samples)
    return {
        "claims": len(ids),
        "replicates": replicates,
        "mean": statistics.fmean(claim_means.values()),
        "lower_95": ordered[int(0.025 * replicates)],
        "upper_95": ordered[min(replicates - 1, math.ceil(0.975 * replicates) - 1)],
        "positive_claim_fraction": statistics.fmean(
            value > 0 for value in claim_means.values()
        ),
    }


def factual_metrics(scores: list[float], labels: list[int]) -> dict[str, float]:
    probabilities_true = [
        1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, score))))
        for score in scores
    ]
    predictions = [int(value >= 0.5) for value in probabilities_true]
    accuracy = statistics.fmean(
        int(prediction == label)
        for prediction, label in zip(predictions, labels)
    )
    brier = statistics.fmean(
        (probability - label) ** 2
        for probability, label in zip(probabilities_true, labels)
    )
    nll = statistics.fmean(
        -math.log(
            max(
                1e-12,
                probability if label == 1 else 1.0 - probability,
            )
        )
        for probability, label in zip(probabilities_true, labels)
    )
    confidences = [
        max(probability, 1.0 - probability)
        for probability in probabilities_true
    ]
    correctness = [
        int(prediction == label)
        for prediction, label in zip(predictions, labels)
    ]
    ece = 0.0
    for lower in (0.5, 0.6, 0.7, 0.8, 0.9):
        upper = lower + 0.1
        indices = [
            index for index, confidence in enumerate(confidences)
            if lower <= confidence < upper or (
                upper >= 1.0 and confidence == 1.0
            )
        ]
        if indices:
            ece += len(indices) / len(scores) * abs(
                statistics.fmean(confidences[index] for index in indices)
                - statistics.fmean(correctness[index] for index in indices)
            )
    correct_margins = [
        score if label == 1 else -score
        for score, label in zip(scores, labels)
    ]
    return {
        "accuracy": accuracy,
        "brier": brier,
        "negative_log_likelihood": nll,
        "expected_calibration_error_5_bins": ece,
        "mean_correct_log_odds_margin": statistics.fmean(correct_margins),
    }


def factual_claim_bootstrap(
    scores: list[float],
    labels: list[int],
    claim_ids: list[int],
    seed: int,
    replicates: int = 2000,
) -> dict[str, dict[str, float]]:
    by_claim: dict[int, list[int]] = {}
    for index, claim_id in enumerate(claim_ids):
        by_claim.setdefault(claim_id, []).append(index)
    ids = sorted(by_claim)
    generator = random.Random(seed)
    sampled_metrics: dict[str, list[float]] = {
        "accuracy": [], "brier": [], "mean_correct_log_odds_margin": []
    }
    for _ in range(replicates):
        sampled_ids = [ids[generator.randrange(len(ids))] for _ in ids]
        indices = [
            index for claim_id in sampled_ids for index in by_claim[claim_id]
        ]
        metrics = factual_metrics(
            [scores[index] for index in indices],
            [labels[index] for index in indices],
        )
        for name in sampled_metrics:
            sampled_metrics[name].append(metrics[name])
    output = {}
    point = factual_metrics(scores, labels)
    for name, values in sampled_metrics.items():
        ordered = sorted(values)
        output[name] = {
            "estimate": point[name],
            "lower_95": ordered[int(0.025 * replicates)],
            "upper_95": ordered[min(
                replicates - 1, math.ceil(0.975 * replicates) - 1
            )],
        }
    return output


def fit_curve(doses: list[float], values: list[float]) -> dict[str, Any]:
    x = torch.tensor(doses, dtype=torch.float64)
    y = torch.tensor(values, dtype=torch.float64)

    def solve(design: torch.Tensor) -> tuple[list[float], float]:
        parameters = torch.linalg.lstsq(design, y).solution
        residual = y - design @ parameters
        return [float(value) for value in parameters], float(
            residual.square().sum().item()
        )

    linear_parameters, linear_sse = solve(
        torch.stack((torch.ones_like(x), x), dim=1)
    )
    best_hill = None
    best_biphasic = None
    for hill in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        for ec50 in torch.logspace(-2.0, 0.3, 48, dtype=torch.float64):
            saturation = x.pow(hill) / (
                ec50.pow(hill) + x.pow(hill)
            ).clamp_min(1e-30)
            hill_parameters, hill_sse = solve(
                torch.stack((torch.ones_like(x), saturation), dim=1)
            )
            candidate = {
                "parameters": {
                    "E0": hill_parameters[0],
                    "Emax": hill_parameters[1],
                    "EC50": float(ec50.item()),
                    "hill": hill,
                },
                "sse": hill_sse,
            }
            if best_hill is None or hill_sse < best_hill["sse"]:
                best_hill = candidate
            biphasic_parameters, biphasic_sse = solve(
                torch.stack(
                    (torch.ones_like(x), saturation, -x.square()), dim=1
                )
            )
            # The specified biphasic model subtracts gamma*D^2; gamma must be
            # non-negative. Reject unconstrained fits that turn this term into
            # an additional positive quadratic.
            if biphasic_parameters[2] < 0:
                continue
            candidate_biphasic = {
                "parameters": {
                    "E0": biphasic_parameters[0],
                    "Emax": biphasic_parameters[1],
                    "EC50": float(ec50.item()),
                    "hill": hill,
                    "gamma": biphasic_parameters[2],
                },
                "sse": biphasic_sse,
            }
            if (
                best_biphasic is None
                or biphasic_sse < best_biphasic["sse"]
            ):
                best_biphasic = candidate_biphasic
    if best_biphasic is None:
        best_biphasic = {
            "parameters": {
                **best_hill["parameters"],
                "gamma": 0.0,
            },
            "sse": best_hill["sse"],
        }
    models = {
        "linear": {
            "parameters": {
                "E0": linear_parameters[0], "beta": linear_parameters[1]
            },
            "sse": linear_sse,
            "parameter_count": 2,
        },
        "hill": {**best_hill, "parameter_count": 4},
        "biphasic": {**best_biphasic, "parameter_count": 5},
    }
    for model in models.values():
        model["aic"] = (
            len(values) * math.log(max(model["sse"] / len(values), 1e-30))
            + 2 * model["parameter_count"]
        )
    return {
        "models": models,
        "best_by_aic": min(models, key=lambda name: models[name]["aic"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoints", type=Path)
    parser.add_argument("basis", type=Path)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--causal-dose", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu-memory", default="14500MiB")
    parser.add_argument("--cpu-memory", default="12GiB")
    parser.add_argument(
        "--offload-folder",
        type=Path,
        default=Path("artifacts/_cache/r26_causal_offload"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/pkpd/r26_causal_and_dose_t4.json"),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_model(args)
    device = model.model.embed_tokens.weight.device
    endpoint_bundle = torch.load(
        args.endpoints, map_location="cpu", weights_only=False
    )
    basis_bundle = torch.load(
        args.basis, map_location="cpu", weights_only=False
    )
    endpoints = endpoint_bundle["endpoints"].to(device, torch.float32)
    contrasts = endpoints[..., 1, :] - endpoints[..., 0, :]
    construction = [int(value) for value in endpoint_bundle["claim_split"]["construction"]]
    test = [int(value) for value in endpoint_bundle["claim_split"]["test"]]
    claims = endpoint_bundle["claims"]
    frame_families = [
        str(row["family"]) for row in endpoint_bundle["response_frames"]
    ]
    frame_positions = [
        str(row["position"]) for row in endpoint_bundle["response_frames"]
    ]
    receptor = basis_bundle["mean_receptor"].to(device, torch.float32)
    receptor_perp = basis_bundle["nuisance_projected_receptor"].to(
        device, torch.float32
    )

    # Causal leave-response-family-out transfer on test claims.
    family_transfer = {}
    for family in sorted(set(frame_families)):
        train_frames = [
            index for index, value in enumerate(frame_families)
            if value != family
        ]
        heldout_frames = [
            index for index, value in enumerate(frame_families)
            if value == family
        ]
        direction = normalize(
            contrasts[construction][:, :, train_frames, :].mean(dim=(0, 1, 2))
        )
        rows = []
        for claim_id in test:
            for question in QUESTION_TEMPLATES:
                prompt = question.format(claim=claims[claim_id]["claim"])
                for frame_id in heldout_frames:
                    frame = RESPONSE_FRAMES[frame_id]
                    rows.append((
                        prompt, frame.high, frame.low,
                        {"claim_id": claim_id, "frame_id": frame_id},
                    ))
        examples, pair_metadata = paired_examples(tokenizer, rows)
        max_steps = max(row["answer_length"] for row in examples)
        effects = make_pkpd_effects(max_steps, args.causal_dose)
        baseline, logits = build_baseline_cache(
            model, tokenizer, examples, args.layer, effects, args.batch_size
        )
        family_transfer[family] = evaluate_signed(
            model, tokenizer, examples, baseline, logits, direction, effects,
            args.layer, args.batch_size, pair_metadata,
            args.seed + heldout_frames[0],
        )
        family_transfer[family]["construction_frames"] = train_frames
        family_transfer[family]["heldout_frames"] = heldout_frames
        print(f"causal family transfer: {family}")

    position_transfer = {}
    for heldout_position in ("initial", "final"):
        train_frames = [
            index for index, value in enumerate(frame_positions)
            if value != heldout_position
        ]
        heldout_frames = [
            index for index, value in enumerate(frame_positions)
            if value == heldout_position
        ]
        direction = normalize(
            contrasts[construction][:, :, train_frames, :].mean(dim=(0, 1, 2))
        )
        rows = []
        for claim_id in test:
            prompt = QUESTION_TEMPLATES[0].format(
                claim=claims[claim_id]["claim"]
            )
            for frame_id in heldout_frames:
                frame = RESPONSE_FRAMES[frame_id]
                rows.append((
                    prompt, frame.high, frame.low,
                    {"claim_id": claim_id, "frame_id": frame_id},
                ))
        examples, pair_metadata = paired_examples(tokenizer, rows)
        max_steps = max(row["answer_length"] for row in examples)
        effects = make_pkpd_effects(max_steps, args.causal_dose)
        baseline, logits = build_baseline_cache(
            model, tokenizer, examples, args.layer, effects, args.batch_size
        )
        position_transfer[heldout_position] = evaluate_signed(
            model, tokenizer, examples, baseline, logits, direction, effects,
            args.layer, args.batch_size, pair_metadata,
            args.seed + 100 + heldout_frames[0],
        )

    # Fresh endpoint rows use test claims only and no construction response text.
    language_rows = []
    factual_rows = []
    decision_rows = []
    numeric_rows = []
    factual_labels = []
    for claim_id in test:
        claim = claims[claim_id]
        language_prompt = (
            f"Evaluate this claim carefully: {claim['claim']}."
        )
        for pair_index, pair in enumerate(FRESH_LANGUAGE_PAIRS):
            language_rows.append((
                language_prompt, pair[0], pair[1],
                {"claim_id": claim_id, "pair_index": pair_index},
            ))
        for statement, label in (
            (str(claim["claim"]), 1),
            (str(claim["false_claim"]), 0),
        ):
            factual_rows.append((
                f"Is the following statement true or false? {statement}",
                TRUE_FALSE_PAIR[0], TRUE_FALSE_PAIR[1],
                {"claim_id": claim_id, "label": label},
            ))
            factual_labels.append(label)
            decision_rows.append((
                f"Can you make a definite judgment about this statement? {statement}",
                ANSWER_ABSTAIN_PAIR[0], ANSWER_ABSTAIN_PAIR[1],
                {"claim_id": claim_id, "label": label},
            ))
            numeric_rows.append((
                f"Judge this statement and report confidence: {statement}",
                NUMERIC_CONFIDENCE_PAIR[0], NUMERIC_CONFIDENCE_PAIR[1],
                {"claim_id": claim_id, "label": label},
            ))
    endpoint_rows = {
        "language": language_rows,
        "decision_confidence": decision_rows,
        "numeric_confidence": numeric_rows,
        "factual_true_false": factual_rows,
    }
    assets = {}
    for name, rows in endpoint_rows.items():
        examples, metadata = paired_examples(tokenizer, rows)
        max_steps = max(row["answer_length"] for row in examples)
        reference_effects = make_pkpd_effects(max_steps, max(DOSES))
        baseline, logits = build_baseline_cache(
            model, tokenizer, examples, args.layer, reference_effects,
            args.batch_size,
        )
        assets[name] = {
            "examples": examples,
            "metadata": metadata,
            "max_steps": max_steps,
            "baseline": baseline,
            "logits": logits,
        }

    dose_rows = []
    projected_comparison = {}
    for dose in DOSES:
        row: dict[str, Any] = {"dose": dose, "endpoints": {}}
        kl_values = []
        for name, asset in assets.items():
            if dose == 0:
                result = asset["baseline"]
            else:
                effects = make_pkpd_effects(asset["max_steps"], dose)
                result = evaluate_direction(
                    model, tokenizer, asset["examples"], args.layer, effects,
                    receptor, 1.0, asset["baseline"], asset["logits"],
                    args.batch_size,
                )
            if name == "factual_true_false":
                metrics = factual_metrics(result.scores, factual_labels)
                metrics["claim_block_bootstrap"] = factual_claim_bootstrap(
                    result.scores,
                    factual_labels,
                    [int(value["claim_id"]) for value in asset["metadata"]],
                    seed=args.seed + int(round(dose * 10000)),
                )
                metrics["mean_score_delta"] = mean_delta(
                    result, asset["baseline"]
                )
            else:
                score_deltas = [
                    score - baseline
                    for score, baseline in zip(
                        result.scores, asset["baseline"].scores
                    )
                ]
                metrics = {
                    "mean_pair_log_odds": statistics.fmean(result.scores),
                    "mean_score_delta": mean_delta(result, asset["baseline"]),
                    "claim_block_bootstrap": grouped_bootstrap(
                        score_deltas,
                        [int(value["claim_id"]) for value in asset["metadata"]],
                        seed=args.seed + int(round(dose * 10000)),
                    ),
                }
            metrics["kl"] = result.kl_auc
            kl_values.append(result.kl_auc)
            row["endpoints"][name] = metrics
        row["mean_kl"] = statistics.fmean(kl_values)
        dose_rows.append(row)
        print(f"dose {dose:g}")

    # Raw versus nuisance-projected receptor at a representative dose.
    for name, direction in (
        ("raw_mean_receptor", receptor),
        ("nuisance_projected_receptor", receptor_perp),
    ):
        endpoint_results = {}
        for endpoint_name, asset in assets.items():
            effects = make_pkpd_effects(asset["max_steps"], args.causal_dose)
            result = evaluate_direction(
                model, tokenizer, asset["examples"], args.layer, effects,
                direction, 1.0, asset["baseline"], asset["logits"],
                args.batch_size,
            )
            endpoint_results[endpoint_name] = {
                "mean_score_delta": mean_delta(result, asset["baseline"]),
                "kl": result.kl_auc,
            }
            if endpoint_name == "factual_true_false":
                endpoint_results[endpoint_name].update(
                    factual_metrics(result.scores, factual_labels)
                )
        projected_comparison[name] = endpoint_results

    curve_metrics = {
        "language_effect": [
            row["endpoints"]["language"]["mean_score_delta"] for row in dose_rows
        ],
        "decision_confidence_effect": [
            row["endpoints"]["decision_confidence"]["mean_score_delta"]
            for row in dose_rows
        ],
        "numeric_confidence_effect": [
            row["endpoints"]["numeric_confidence"]["mean_score_delta"]
            for row in dose_rows
        ],
        "accuracy": [
            row["endpoints"]["factual_true_false"]["accuracy"] for row in dose_rows
        ],
        "brier": [
            row["endpoints"]["factual_true_false"]["brier"] for row in dose_rows
        ],
        "calibration_ece": [
            row["endpoints"]["factual_true_false"][
                "expected_calibration_error_5_bins"
            ]
            for row in dose_rows
        ],
        "mean_kl": [row["mean_kl"] for row in dose_rows],
    }
    curve_fits = {
        name: fit_curve(list(DOSES), values)
        for name, values in curve_metrics.items()
    }
    family_effects = [
        row["antisymmetric_effect"] for row in family_transfer.values()
    ]
    payload = {
        "experiment": "R2.6_unseen_template_transfer_and_mean_receptor_dose_response",
        "model": MODEL_NAME,
        "layer": args.layer,
        "site": "block_input_residual_pre",
        "tolerance_or_homeostasis_enabled": False,
        "unseen_response_family_causal_transfer": family_transfer,
        "unseen_phrase_position_causal_transfer": position_transfer,
        "causal_transfer_summary": {
            "families_positive": sum(value > 0 for value in family_effects),
            "families_tested": len(family_effects),
            "mean_antisymmetric_effect": statistics.fmean(family_effects),
            "minimum_antisymmetric_effect": min(family_effects),
        },
        "dose_response": dose_rows,
        "curve_fits": curve_fits,
        "raw_vs_nuisance_projected_at_causal_dose": projected_comparison,
        "fresh_endpoint_notes": {
            "language": "Four response pairs absent from receptor construction.",
            "decision_confidence": "Definite-answer versus abstention sequence log odds.",
            "numeric_confidence": "90 percent versus 50 percent confidence sequence log odds.",
            "accuracy_and_calibration": "True/false matched statements with probability derived from paired sequence log odds.",
        },
        "cuda": {
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
        "elapsed_seconds": time.time() - start,
        "limitations": [
            "Teacher-forced candidate probabilities are normalized over two response sequences, not the model's full open-ended answer distribution.",
            "Factual test items are synthetic matched true/false statements and do not constitute a broad knowledge benchmark.",
            "Causal family-transfer cells share test claims and are not independent replications.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
