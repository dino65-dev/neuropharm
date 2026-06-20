"""Produce a strict statistical and geometry verdict for R2."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path


def bootstrap_ci(values: list[float], seed: int, samples: int = 20_000) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(samples)
    )
    return [estimates[int(0.025 * samples)], estimates[int(0.975 * samples)]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    output = args.output or args.input.with_name(args.input.stem + "_analysis.json")
    receptor = payload["receptor"]
    random_controls = payload["random_null"]["controls"]
    random_effects = [
        control["test_antisymmetric_effect"] for control in random_controls
    ]
    random_kls = [control["test_kl_auc"] for control in random_controls]
    receptor_kl = receptor["test_kl_auc"]
    mean_random_kl = statistics.fmean(random_kls)
    kl_ratio = mean_random_kl / receptor_kl

    # Conservative first-order sensitivity check: rescale every random effect
    # by the factor needed to match receptor test KL under local KL~dose^2 and
    # effect~dose assumptions.
    conservative_effects = [
        effect * math.sqrt(receptor_kl / max(kl, 1e-12))
        for effect, kl in zip(random_effects, random_kls)
    ]
    conservative_q95 = sorted(conservative_effects)[
        math.ceil(0.95 * len(conservative_effects)) - 1
    ]
    geometry = payload["geometry"]
    raw_ev = geometry["raw_spectrum"]["explained_variance"]
    centered_ev = geometry["centered_spectrum"]["explained_variance"]
    if raw_ev["1"] >= 0.8 and centered_ev["1"] >= 0.5:
        geometry_case = "A_coherent_rank_one"
    elif centered_ev["4"] >= 0.75 and geometry["coherence"] >= 0.5:
        geometry_case = "B_coherent_low_rank"
    else:
        geometry_case = "C_high_rank_or_context_dependent"

    analysis = {
        "source": str(args.input),
        "gpu": {
            "device": payload["cuda"]["device"],
            "capability": payload["cuda"]["capability"],
            "peak_allocated_gib": payload["cuda"]["peak_allocated_bytes"] / 2**30,
            "peak_reserved_gib": payload["cuda"]["peak_reserved_bytes"] / 2**30,
            "elapsed_hours": payload["elapsed_seconds"] / 3600,
        },
        "data_integrity": {
            "claim_group_disjoint": payload["gates"]["claim_group_disjoint"],
            "construction_claims": len(payload["claim_group_split"]["construction"]),
            "validation_claims": len(payload["claim_group_split"]["validation"]),
            "fresh_test_claims": len(payload["claim_group_split"]["test"]),
            "evaluation_response_pairs": len(payload["evaluation_response_pairs"]),
            "response_lengths_matched": payload["gates"]["response_lengths_matched"],
        },
        "geometry": {
            "case": geometry_case,
            "coherence": geometry["coherence"],
            "mean_pairwise_cosine": geometry["mean_pairwise_cosine_from_coherence"],
            "raw_spectrum": geometry["raw_spectrum"],
            "centered_spectrum": geometry["centered_spectrum"],
            "claim_grouped_split_half": geometry["claim_grouped_split_half"],
            "claim_bootstrap": geometry["claim_bootstrap"],
            "leave_one_claim_out": geometry["leave_one_claim_out"],
            "leave_one_domain_out": geometry["leave_one_domain_out"],
            "leave_one_construction_response_pair_out": (
                geometry["leave_one_construction_response_pair_out"]
            ),
        },
        "causal_test": {
            "receptor_validation_positive_delta": receptor["validation_positive_delta"],
            "receptor_validation_negative_delta": receptor["validation_negative_delta"],
            "receptor_test_positive_delta": receptor["test_positive_delta"],
            "receptor_test_negative_delta": receptor["test_negative_delta"],
            "receptor_test_antisymmetric_effect": receptor["test_antisymmetric_effect"],
            "random_count": len(random_effects),
            "random_q95": payload["random_null"]["q95"],
            "random_exceedances": payload["random_null"]["receptor_exceedances"],
            "empirical_p_plus_one": payload["random_null"]["empirical_p_plus_one"],
            "monte_carlo_statement": (
                f"The receptor exceeded "
                f"{len(random_effects) - payload['random_null']['receptor_exceedances']}"
                f"/{len(random_effects)} matched random controls; empirical "
                f"p={payload['random_null']['empirical_p_plus_one']:.4f}."
            ),
            "conservative_test_kl_rescaled_random_q95": conservative_q95,
        },
        "disturbance_matching": {
            "receptor_test_kl_auc": receptor_kl,
            "mean_random_test_kl_auc": mean_random_kl,
            "mean_random_to_receptor_kl_ratio": kl_ratio,
            "random_test_kl_min": min(random_kls),
            "random_test_kl_max": max(random_kls),
            "validation_matching_method": (
                "empirical unit-dose KL followed by local quadratic scaling"
            ),
            "test_distribution_tolerance_20pct_passed": 0.8 <= kl_ratio <= 1.2,
        },
        "gates": {
            "signed_effect": payload["gates"]["sign"],
            "random_null": payload["gates"]["random_null"],
            "conservative_random_null_after_test_kl_sensitivity": (
                receptor["test_antisymmetric_effect"] > conservative_q95
            ),
            "disturbance_match_within_20pct": 0.8 <= kl_ratio <= 1.2,
            "claim_grouped_split_half_identifiable": (
                geometry["claim_grouped_split_half"]["p05"] > 0.8
            ),
            "claim_bootstrap_identifiable": (
                geometry["claim_bootstrap"]["p05"] > 0.8
            ),
            "leave_one_claim_stable": (
                geometry["leave_one_claim_out"]["minimum"] > 0.8
            ),
            "leave_one_domain_stable": (
                geometry["leave_one_domain_out"]["minimum"] > 0.8
            ),
            "leave_one_response_pair_stable": (
                geometry["leave_one_construction_response_pair_out"]["minimum"]
                > 0.8
            ),
        },
    }
    analysis["overall_passed"] = all(analysis["gates"].values())
    analysis["decision"] = (
        "Proceed to low-rank receptor-subspace and gradient-alignment candidates; "
        "do not enable tolerance until a selected candidate replicates."
        if analysis["overall_passed"]
        else "Repair failed gates before receptor optimization."
    )
    output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
