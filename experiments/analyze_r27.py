"""Issue the R2.7 receptor-selection verdict without test-set tuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/pkpd/r27_research_verdict_t4.json"),
    )
    args = parser.parse_args()
    source = json.loads(args.artifact.read_text(encoding="utf-8"))
    selection = source["validation_selection"]
    selected = selection["selected"]
    validation = source["validation_phenotypes"]
    test = source["untouched_test_phenotypes"]
    raw_test = test["lambda_0.00"]
    projected_test = test["lambda_1.00"]
    nuisance_test = test["nuisance_component"]
    payload = {
        "experiment": "R2.7_research_verdict",
        "source": str(args.artifact),
        "kl_matching": {
            "target": source["common_validation_kl_target"],
            "dose_range": [
                min(row["dose"] for row in source["dose_matching"].values()),
                max(row["dose"] for row in source["dose_matching"].values()),
            ],
            "maximum_absolute_matching_error": max(
                row["absolute_error"]
                for row in source["dose_matching"].values()
            ),
        },
        "validation": {
            "pareto_front": source["validation_lambda_pareto_front"],
            "selected_variant": selected,
            "feasible_variants": selection["feasible"],
            "selection_rule": selection["rule"],
            "selected_phenotype": validation[selected],
        },
        "untouched_test": {
            "selected_variant": test[selected],
            "raw": raw_test,
            "fully_projected": projected_test,
            "nuisance_component": nuisance_test,
        },
        "causal_phenotype_decomposition": {
            "nuisance_component_language_delta": nuisance_test["language_delta"],
            "nuisance_component_definite_delta": nuisance_test[
                "definite_answer_delta"
            ],
            "nuisance_component_numeric_delta": nuisance_test[
                "numeric_confidence_delta"
            ],
            "projected_over_raw_language_ratio": (
                projected_test["language_delta"] / raw_test["language_delta"]
            ),
            "projected_over_raw_definite_ratio": (
                projected_test["definite_answer_delta"]
                / raw_test["definite_answer_delta"]
            ),
            "projected_over_raw_numeric_ratio": (
                projected_test["numeric_confidence_delta"]
                / raw_test["numeric_confidence_delta"]
            ),
        },
        "verdict": (
            "No partially nuisance-attenuated receptor dominates the raw "
            "direction at matched validation KL. The nuisance component has a "
            "style/definite-answer phenotype and lowers numeric confidence; "
            "the fully projected component raises numeric confidence but also "
            "raises confidence on baseline-wrong items. Retain the raw fixed "
            "receptor for the context-gain diagnostic; do not interpret the "
            "geometric projection as a clean causal decomposition."
        ),
        "next_gate": (
            "Estimate claim-level local causal slopes and KL curvature. Train "
            "a context-dependent gain only if sign is sufficiently stable and "
            "efficiency varies predictably from pre-intervention features."
        ),
        "enable_tolerance_or_homeostasis": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
