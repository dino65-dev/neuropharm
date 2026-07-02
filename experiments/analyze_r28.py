"""Combine R2.8 local-slope and controller evidence into a gate verdict."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compact(metrics):
    keys = (
        "mean_absolute_language_delta",
        "definite_answer_delta",
        "numeric_confidence_delta",
        "numeric_confidence_delta_baseline_incorrect",
        "answer_margin_delta",
        "accuracy_delta",
        "brier_delta",
        "nll_delta",
        "mean_kl",
    )
    return {key: metrics[key] for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("r27_verdict", type=Path)
    parser.add_argument("slopes", type=Path)
    parser.add_argument("controller", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/pkpd/r28_research_verdict_t4.json"),
    )
    args = parser.parse_args()
    r27 = json.loads(args.r27_verdict.read_text(encoding="utf-8"))
    slopes = json.loads(args.slopes.read_text(encoding="utf-8"))
    controller = json.loads(args.controller.read_text(encoding="utf-8"))
    test = controller["untouched_test_metrics"]
    gate = test["gated"]
    gate_fixed = test["kl_matched_fixed_for_gated"]
    continuous = test["continuous"]
    continuous_fixed = test["kl_matched_fixed_for_continuous"]
    oracle = test["oracle_upper_bound"]
    predictor_correlation = controller["controller_training"][
        "tune_prediction_correlation_with_definite_slope"
    ]

    gate_beats_fixed = (
        gate["definite_answer_delta"] >= gate_fixed["definite_answer_delta"]
        and gate["numeric_confidence_delta_baseline_incorrect"]
        <= gate_fixed["numeric_confidence_delta_baseline_incorrect"]
        and gate["answer_margin_delta"] >= gate_fixed["answer_margin_delta"]
    )
    continuous_beats_fixed = (
        continuous["definite_answer_delta"]
        >= continuous_fixed["definite_answer_delta"]
        and continuous["numeric_confidence_delta_baseline_incorrect"]
        <= continuous_fixed["numeric_confidence_delta_baseline_incorrect"]
        and continuous["answer_margin_delta"]
        >= continuous_fixed["answer_margin_delta"]
    )
    payload = {
        "experiment": "R2.8_combined_research_verdict",
        "sources": {
            "r27": str(args.r27_verdict),
            "slopes": str(args.slopes),
            "controller": str(args.controller),
        },
        "receptor_variant": {
            "selected": r27["validation"]["selected_variant"],
            "verdict": r27["verdict"],
        },
        "local_causal_geometry": {
            "gain_gate": slopes["gain_gate"],
            "definite_beta": slopes["aggregated_across_epsilon"][
                "definite"
            ]["beta"],
            "numeric_beta": slopes["aggregated_across_epsilon"][
                "numeric"
            ]["beta"],
            "factual_beta_baseline_incorrect": slopes[
                "aggregated_across_epsilon"
            ]["factual"]["beta_by_baseline_correctness"]["incorrect"],
            "interpretation": (
                "The fixed receptor has a highly consistent positive local "
                "effect on definite-answer preference, but factual-margin "
                "effects vary in sign and are negative on average for "
                "baseline-wrong items."
            ),
        },
        "controller_predictability": {
            "heldout_validation_correlation": predictor_correlation,
            "passed": predictor_correlation > 0.1,
            "interpretation": (
                "The tested baseline margin, entropy, endpoint-score, and "
                "prompt-length features do not predict local efficacy."
            ),
        },
        "untouched_test": {
            "gated": compact(gate),
            "validation_kl_matched_fixed_for_gate": compact(gate_fixed),
            "gate_beats_fixed_safety_and_efficacy": gate_beats_fixed,
            "continuous": compact(continuous),
            "validation_kl_matched_fixed_for_continuous": compact(
                continuous_fixed
            ),
            "continuous_beats_fixed_safety_and_efficacy": (
                continuous_beats_fixed
            ),
            "oracle_upper_bound": compact(oracle),
        },
        "verdict": (
            "Context-dependent gain is not validated. Local slopes show that "
            "selective dosing could help, and the oracle confirms substantial "
            "headroom, but the current pre-intervention feature set predicts "
            "slope with the wrong sign and both learned policies lose to "
            "validation-KL-matched fixed controls on the joint efficacy and "
            "baseline-wrong-confidence criteria."
        ),
        "next_experiment": (
            "Do not deploy adaptive gain. Add semantic uncertainty and "
            "prompt-residual features, use cross-fitted policy training, and "
            "repeat the matched-KL gate test. Run the adverse-event panel "
            "before any policy is considered deployable."
        ),
        "research_decision": {
            "use_centered_subspace": False,
            "use_nuisance_projected_receptor": False,
            "validated_receptor": "raw_mean_receptor",
            "validated_controller": None,
            "default_policy": "zero_or_fixed_dose_only_in_controlled_experiments",
            "enable_signed_dosing": False,
            "enable_tolerance_or_homeostasis": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
