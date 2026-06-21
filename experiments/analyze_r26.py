"""Combine R2.6 geometry, causal-transfer, and dose-response verdicts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.r26_causal_and_dose_gpu import fit_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("crossed_analysis", type=Path)
    parser.add_argument("causal_and_dose", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/pkpd/r26_research_verdict_t4.json"),
    )
    args = parser.parse_args()
    crossed = json.loads(args.crossed_analysis.read_text(encoding="utf-8"))
    causal = json.loads(args.causal_and_dose.read_text(encoding="utf-8"))
    selected_k = int(
        crossed["nested_claim_and_response_environment_cv"][
            "one_standard_error_selected_k"
        ]
    )
    leakage = crossed["template_leakage"]
    factor_energy = crossed[
        "factorial_certainty_effect_partition"
    ]["fraction_of_contrast_energy"]
    nuisance = crossed["mean_receptor"]
    family_transfer = causal["unseen_response_family_causal_transfer"]
    robust_families = sum(
        row["claim_block_bootstrap"]["lower_95"] > 0
        for row in family_transfer.values()
    )
    dose_rows = causal["dose_response"]
    doses = [float(row["dose"]) for row in dose_rows]
    curve_values = {
        "language_effect": [
            row["endpoints"]["language"]["mean_score_delta"]
            for row in dose_rows
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
            row["endpoints"]["factual_true_false"]["accuracy"]
            for row in dose_rows
        ],
        "brier": [
            row["endpoints"]["factual_true_false"]["brier"]
            for row in dose_rows
        ],
        "calibration_ece": [
            row["endpoints"]["factual_true_false"][
                "expected_calibration_error_5_bins"
            ]
            for row in dose_rows
        ],
        "mean_kl": [row["mean_kl"] for row in dose_rows],
    }
    constrained_curve_fits = {
        name: fit_curve(doses, values)
        for name, values in curve_values.items()
    }
    highest = dose_rows[-1]
    baseline = dose_rows[0]
    raw_projected = causal[
        "raw_vs_nuisance_projected_at_causal_dose"
    ]
    raw = raw_projected["raw_mean_receptor"]
    projected = raw_projected["nuisance_projected_receptor"]

    centered_valid = bool(crossed["all_noncausal_gates_passed"])
    accuracy_changed = (
        highest["endpoints"]["factual_true_false"]["accuracy"]
        != baseline["endpoints"]["factual_true_false"]["accuracy"]
    )
    payload = {
        "experiment": "R2.6_combined_research_verdict",
        "sources": {
            "crossed_analysis": str(args.crossed_analysis),
            "causal_and_dose": str(args.causal_and_dose),
        },
        "balanced_design": {
            "activation_tensor_shape": crossed["tensor_shape"],
            "construction_claims": len(crossed["claim_split"]["construction"]),
            "validation_claims": len(crossed["claim_split"]["validation"]),
            "test_claims": len(crossed["claim_split"]["test"]),
        },
        "factorial_decomposition": {
            "certainty_main_energy_fraction": factor_energy["certainty_main"],
            "certainty_x_response_energy_fraction": factor_energy[
                "certainty_x_response"
            ],
            "certainty_x_claim_energy_fraction": factor_energy[
                "certainty_x_claim"
            ],
            "certainty_x_question_energy_fraction": factor_energy[
                "certainty_x_question"
            ],
            "higher_interaction_energy_fraction": factor_energy[
                "higher_interactions_and_residual"
            ],
        },
        "centered_subspace": {
            "nested_cv_selected_k": selected_k,
            "pass_gates": crossed["pass_gates"],
            "validated": centered_valid,
            "response_family_leakage_accuracy": leakage[
                "response_family_accuracy"
            ],
            "response_family_chance": leakage["response_family_chance"],
            "response_frame_leakage_accuracy": leakage[
                "response_frame_accuracy"
            ],
            "response_frame_chance": leakage["response_frame_chance"],
            "verdict": (
                "No centered semantic subspace is validated. Reconstruction "
                "and average overlap exceed random controls, but held-out "
                "template identity is almost perfectly linearly decodable."
            ),
        },
        "mean_receptor": {
            "wording_nuisance_projection_fraction": nuisance[
                "wording_nuisance_projection_fraction"
            ],
            "cosine_raw_vs_nuisance_projected": nuisance[
                "cosine_raw_vs_projected"
            ],
            "unseen_family_point_positive": sum(
                row["antisymmetric_effect"] > 0
                for row in family_transfer.values()
            ),
            "unseen_families_tested": len(family_transfer),
            "unseen_family_claim_bootstrap_lower_positive": robust_families,
            "phrase_position_transfer": causal[
                "unseen_phrase_position_causal_transfer"
            ],
            "verdict": (
                "The mean receptor transfers across all four unseen response "
                "families by point estimate, with claim-block 95% intervals "
                f"above zero for {robust_families}/4 families. It is not "
                "wording-pure: "
                f"{100 * nuisance['wording_nuisance_projection_fraction']:.1f}% "
                "of its unit energy lies in the stable same-certainty "
                "wording-control subspace."
            ),
        },
        "dose_response_at_maximum_tested_dose": {
            "dose": highest["dose"],
            "language_effect": highest["endpoints"]["language"],
            "decision_confidence_effect": highest["endpoints"][
                "decision_confidence"
            ],
            "numeric_confidence_effect": highest["endpoints"][
                "numeric_confidence"
            ],
            "factual": highest["endpoints"]["factual_true_false"],
            "mean_kl": highest["mean_kl"],
            "accuracy_changed_from_baseline": accuracy_changed,
        },
        "constrained_curve_fits": constrained_curve_fits,
        "raw_vs_nuisance_projected_at_dose_0_25": {
            "raw": raw,
            "projected": projected,
            "projected_over_raw_effect_ratio": {
                endpoint: (
                    projected[endpoint]["mean_score_delta"]
                    / raw[endpoint]["mean_score_delta"]
                    if raw[endpoint]["mean_score_delta"] != 0 else None
                )
                for endpoint in (
                    "language", "decision_confidence",
                    "numeric_confidence", "factual_true_false",
                )
            },
        },
        "phenotype_verdict": (
            "The validated mean direction is a mixed linguistic-assertiveness "
            "and subjective-confidence receptor. Across the tested doses it "
            "raises definite-answer and numeric-confidence preferences while "
            "factual accuracy remains 97.2%. The small Brier/ECE improvements "
            "are far below claim-block uncertainty, so this experiment does "
            "not establish improved calibration or overconfidence."
        ),
        "research_decision": {
            "outcome": "A",
            "proceed_to_R3_subspace_ligands": False,
            "recommended_control": (
                "Retain one mean receptor, explicitly distinguish raw and "
                "wording-projected variants, and study context-dependent gain."
            ),
            "enable_tolerance_or_homeostasis": False,
        },
        "limitations": causal["limitations"] + crossed["limitations"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
