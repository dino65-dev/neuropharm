"""Final R2.7-R2.9 gate verdict."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("r28_verdict", type=Path)
    parser.add_argument("externality", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/pkpd/r29_research_verdict_t4.json"),
    )
    args = parser.parse_args()
    r28 = json.loads(args.r28_verdict.read_text(encoding="utf-8"))
    externality = json.loads(args.externality.read_text(encoding="utf-8"))
    by_alpha = {
        float(row["alpha"]): row for row in externality["rows"]
    }
    retained_alpha = 0.125
    retained = by_alpha[retained_alpha]
    first_adverse = externality["first_adverse_alpha"]
    payload = {
        "experiment": "R2.7_R2.9_final_gate_verdict",
        "sources": {
            "controller_verdict": str(args.r28_verdict),
            "externality_panel": str(args.externality),
        },
        "validated_structure": {
            "receptor": "raw_mean_epistemic_assertiveness_direction",
            "centered_semantic_subspace": False,
            "nuisance_projected_replacement": False,
            "context_dependent_controller": False,
        },
        "externality_panel": {
            "retained_experimental_alpha": retained_alpha,
            "retained_alpha_passed": retained["passes_externality_gate"],
            "retained_alpha_flags": retained["adverse_flags"],
            "first_adverse_alpha": first_adverse,
            "first_adverse_flags": (
                by_alpha[float(first_adverse)]["adverse_flags"]
                if first_adverse is not None else None
            ),
            "interpretation": (
                "The small synthetic panel detects no configured adverse "
                "threshold at alpha <= 0.25. At alpha 0.5, harmful-compliance "
                "sequence log odds cross the preregistered exploratory "
                "threshold. This is a safety signal, not a deployment limit."
            ),
        },
        "controller_verdict": r28["verdict"],
        "overall_verdict": (
            "Retain the raw receptor only as a controlled experimental "
            "intervention. Use fixed direct-additive alpha <= 0.125 for the "
            "next replication. The tested nuisance attenuation and learned "
            "context-gain policies are not validated. Do not use centered "
            "ligands, signed dosing, tolerance, or homeostasis."
        ),
        "next_required_work": [
            "Replicate the raw fixed receptor and externality panel on fresh claims and a second model.",
            "Add semantic-uncertainty and prompt-residual features before retraining any gain controller.",
            "Use cross-fitted policy training and validation-KL-matched fixed comparators.",
            "Replace the synthetic harmful-compliance panel with established safety and jailbreak benchmarks.",
            "Replicate a behavioral dose-response curve before estimating a therapeutic index.",
        ],
        "tolerance_gate": {
            "enabled": False,
            "failed_requirements": [
                "context-dependent gain did not beat matched fixed dosing",
                "incorrect-answer confidence increased under tested policies",
                "externality evidence is synthetic and single-model",
                "behavioral dose-response has not replicated",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
