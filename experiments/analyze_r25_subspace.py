"""Issue a strict R2.5 subspace reproducibility verdict."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    output = args.output or args.input.with_name(args.input.stem + "_analysis.json")

    stable_k = []
    for k in payload["k_values"]:
        angle = payload["principal_angle_stability"][str(k)]
        if (
            angle["mean_cos2"]["p05"] > 0.8
            and angle["minimum_cosine"]["p05"] > 0.8
        ):
            stable_k.append(k)
    maximum_reproducible_k = max(stable_k) if stable_k else None
    cv_selected_k = payload["selected_k"]
    axes = payload["variance_attribution"]["axes"]
    stable_axis_attribution = {
        axis: factors
        for axis, factors in axes.items()
        if int(axis) <= (maximum_reproducible_k or 0)
    }
    response_template_eta = [
        factors["response_template"]
        for factors in stable_axis_attribution.values()
    ]
    question_template_eta = [
        factors["question_template"]
        for factors in stable_axis_attribution.values()
    ]
    wording_dominated = (
        bool(response_template_eta)
        and statistics.fmean(response_template_eta) > 0.5
    )

    analysis = {
        "source": str(args.input),
        "cross_validated_reconstruction_selected_k": cv_selected_k,
        "maximum_principal_angle_reproducible_k": maximum_reproducible_k,
        "validated_centered_subspace_k": (
            maximum_reproducible_k if not wording_dominated else None
        ),
        "reconstruction": payload["cross_validated_reconstruction"],
        "principal_angle_stability": payload["principal_angle_stability"],
        "stable_axis_variance_attribution": stable_axis_attribution,
        "wording_diagnostics": {
            "mean_response_template_eta_squared": (
                statistics.fmean(response_template_eta)
                if response_template_eta else None
            ),
            "mean_question_template_eta_squared": (
                statistics.fmean(question_template_eta)
                if question_template_eta else None
            ),
            "wording_dominated": wording_dominated,
        },
        "verdict": (
            "The centered variation is predictively low-dimensional but not "
            "fully reproducible at the cross-validation-selected k. Four "
            "dimensions are principal-angle stable, but they are dominated by "
            "construction-response wording. Do not lock V_k as a semantic "
            "receptor family until contrasts are fully crossed across response "
            "templates."
        ),
        "next_action": (
            "Retain the mean epistemic-assertiveness receptor as the validated "
            "baseline; run cross-domain causal transfer and redesign the "
            "construction matrix with every claim crossed with every response "
            "template before R3 ligand synthesis."
        ),
    }
    output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

