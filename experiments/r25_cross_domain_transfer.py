"""R2.5 cross-domain causal transfer for the mean receptor family."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.r2_geometry_gpu import (
    CLAIM_DOMAINS,
    EVALUATION_RESPONSE_PAIRS,
    MODEL_NAME,
    antisymmetric_effect,
    assert_response_pairs_are_matched,
    build_baseline_cache,
    evaluate_direction,
    input_device,
    load_model,
    make_evaluation_examples,
    make_pkpd_effects,
    mean_delta,
    prompt_rows,
    split_claim_ids,
)


def normalized_mean(values: torch.Tensor) -> torch.Tensor:
    mean = values.mean(dim=0)
    return mean / mean.norm().clamp_min(1e-12)


def evaluate_receptor(
    model,
    tokenizer,
    examples,
    baseline,
    baseline_logits,
    direction,
    effects,
    layer,
    batch_size,
) -> dict[str, float]:
    positive = evaluate_direction(
        model, tokenizer, examples, layer, effects, direction, 1.0,
        baseline, baseline_logits, batch_size,
    )
    negative = evaluate_direction(
        model, tokenizer, examples, layer, effects, -direction, 1.0,
        baseline, baseline_logits, batch_size,
    )
    return {
        "positive_delta": mean_delta(positive, baseline),
        "negative_delta": mean_delta(negative, baseline),
        "antisymmetric_effect": antisymmetric_effect(
            positive, negative, baseline
        ),
        "kl_auc": (positive.kl_auc + negative.kl_auc) / 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contrast_matrix", type=Path)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--dose", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gpu-memory", default="14500MiB")
    parser.add_argument("--cpu-memory", default="12GiB")
    parser.add_argument(
        "--offload-folder",
        type=Path,
        default=Path("artifacts/_cache/r25_transfer_offload"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/pkpd/r25_cross_domain_transfer.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    start = time.time()
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    pair_lengths = assert_response_pairs_are_matched(tokenizer)
    model = load_model(args)
    bundle = torch.load(
        args.contrast_matrix, map_location="cpu", weights_only=False
    )
    differences = bundle["differences"].to(
        device=input_device(model), dtype=torch.float32
    )
    metadata = bundle["metadata"]
    construction_domains = sorted({str(row["domain"]) for row in metadata})
    full_receptor = normalized_mean(differences)
    domain_receptors = {
        domain: normalized_mean(
            differences[
                [
                    index for index, row in enumerate(metadata)
                    if row["domain"] == domain
                ]
            ]
        )
        for domain in construction_domains
    }
    leave_domain_out_receptors = {
        domain: normalized_mean(
            differences[
                [
                    index for index, row in enumerate(metadata)
                    if row["domain"] != domain
                ]
            ]
        )
        for domain in construction_domains
    }
    construction_domain_claim_counts = Counter(
        str(row["domain"]) for row in metadata[::4]
    )

    split = split_claim_ids(args.seed, 30, 12, 18)
    test_claims_by_domain: dict[str, list[int]] = {}
    for claim_id in split["test"]:
        test_claims_by_domain.setdefault(
            CLAIM_DOMAINS[claim_id], []
        ).append(claim_id)
    effects = make_pkpd_effects(max(pair_lengths), args.dose)
    test_assets: dict[str, dict[str, Any]] = {}
    for domain, claim_ids in sorted(test_claims_by_domain.items()):
        rows = prompt_rows(claim_ids)
        examples = make_evaluation_examples(tokenizer, rows)
        baseline, baseline_logits = build_baseline_cache(
            model, tokenizer, examples, args.layer, effects, args.batch_size
        )
        test_assets[domain] = {
            "claim_ids": claim_ids,
            "rows": rows,
            "examples": examples,
            "baseline": baseline,
            "baseline_logits": baseline_logits,
        }

    full_transfer = {}
    leave_domain_out_transfer = {}
    transfer_matrix: dict[str, dict[str, Any]] = {}
    for target_domain, assets in test_assets.items():
        full_transfer[target_domain] = evaluate_receptor(
            model, tokenizer, assets["examples"], assets["baseline"],
            assets["baseline_logits"], full_receptor, effects,
            args.layer, args.batch_size,
        )
        if target_domain in leave_domain_out_receptors:
            leave_domain_out_transfer[target_domain] = evaluate_receptor(
                model, tokenizer, assets["examples"], assets["baseline"],
                assets["baseline_logits"],
                leave_domain_out_receptors[target_domain], effects,
                args.layer, args.batch_size,
            )

    for source_domain, receptor in domain_receptors.items():
        transfer_matrix[source_domain] = {}
        for target_domain, assets in test_assets.items():
            transfer_matrix[source_domain][target_domain] = evaluate_receptor(
                model, tokenizer, assets["examples"], assets["baseline"],
                assets["baseline_logits"], receptor, effects,
                args.layer, args.batch_size,
            )
        print(
            f"source domain {source_domain}: "
            f"{len(test_assets)} target domains"
        )

    full_effects = [
        row["antisymmetric_effect"] for row in full_transfer.values()
    ]
    leave_out_effects = [
        row["antisymmetric_effect"]
        for row in leave_domain_out_transfer.values()
    ]
    matrix_effects = [
        result["antisymmetric_effect"]
        for targets in transfer_matrix.values()
        for result in targets.values()
    ]
    payload = {
        "experiment": "R2.5_cross_domain_causal_transfer",
        "model": MODEL_NAME,
        "layer": args.layer,
        "site": "block_input_residual_pre",
        "administration_protocol": "single_bolus_no_repeated_pulse",
        "effect_schedule": effects,
        "construction_domain_claim_counts": dict(
            construction_domain_claim_counts
        ),
        "test_domain_claim_ids": test_claims_by_domain,
        "full_receptor_transfer": full_transfer,
        "leave_target_domain_out_transfer": leave_domain_out_transfer,
        "domain_to_domain_transfer_matrix": transfer_matrix,
        "summary": {
            "full_receptor_mean_effect_across_test_domains": statistics.fmean(
                full_effects
            ),
            "full_receptor_positive_domain_fraction": sum(
                value > 0 for value in full_effects
            ) / len(full_effects),
            "leave_domain_out_mean_effect": statistics.fmean(
                leave_out_effects
            ),
            "leave_domain_out_positive_domain_fraction": sum(
                value > 0 for value in leave_out_effects
            ) / len(leave_out_effects),
            "domain_specific_matrix_mean_effect": statistics.fmean(
                matrix_effects
            ),
            "domain_specific_matrix_positive_fraction": sum(
                value > 0 for value in matrix_effects
            ) / len(matrix_effects),
        },
        "cuda": {
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "device_map": model.hf_device_map,
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

