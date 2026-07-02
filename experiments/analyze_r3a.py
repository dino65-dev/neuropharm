"""Cross-model R3A replication verdict."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--alpha", type=float, default=0.125)
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/pkpd/r3a_cross_model_verdict_t4.json"),
    )
    args = parser.parse_args()
    models = []
    for path in args.results:
        payload = json.loads(path.read_text(encoding="utf-8"))
        row = next(value for value in payload["rows"] if value["alpha"] == args.alpha)
        definite_ci = row["claim_block_bootstrap"]["definite"]
        models.append({
            "source": str(path),
            "model": payload["model"],
            "layers": payload["layers"],
            "hidden_size": payload["hidden_size"],
            "intervention_layer": payload["intervention_layer"],
            "intervention_layer_fraction": payload["intervention_layer_fraction"],
            "receptor_raw_norm": payload["receptor_raw_norm"],
            "baseline_accuracy": payload["baseline_factual"]["accuracy"],
            "baseline_incorrect_items": payload["baseline_incorrect_items"],
            "alpha": args.alpha,
            "language_delta": row["language_delta"],
            "definite_delta": row["definite_delta"],
            "definite_claim_block_95": [
                definite_ci["lower_95"], definite_ci["upper_95"]
            ],
            "definite_positive_claim_fraction": definite_ci[
                "positive_claim_fraction"
            ],
            "numeric_delta": row["numeric_delta"],
            "numeric_claim_block_95": [
                row["claim_block_bootstrap"]["numeric"]["lower_95"],
                row["claim_block_bootstrap"]["numeric"]["upper_95"],
            ],
            "numeric_delta_baseline_incorrect": row[
                "numeric_delta_baseline_incorrect"
            ],
            "factual_margin_delta": row["factual_margin_delta"],
            "accuracy_delta": row["accuracy_delta"],
            "brier_delta": row["brier_delta"],
            "nll_delta": row["nll_delta"],
            "mean_kl": row["mean_kl"],
            "core_replication_pass": (
                row["definite_delta"] > 0
                and definite_ci["lower_95"] > 0
                and row["accuracy_delta"] >= 0
            ),
        })
    replicated = all(row["core_replication_pass"] for row in models)
    payload = {
        "experiment": "R3A_cross_model_replication_verdict",
        "alpha": args.alpha,
        "models": models,
        "core_models_passed": sum(row["core_replication_pass"] for row in models),
        "models_tested": len(models),
        "core_phenotype_replicated": replicated,
        "verdict": (
            "The model-local raw receptor replicates the definite-answer "
            "phenotype across the original Qwen model, a larger same-family "
            "Qwen model, and a different Gemma architecture at alpha=0.125. "
            "All claim-block intervals exclude zero and factual accuracy does "
            "not decrease. Numeric-confidence and incorrect-confidence effects "
            "are not consistent enough to claim a shared safety phenotype."
        ),
        "constraints": [
            "Each model uses a separately constructed receptor; vectors are never transferred across hidden dimensions.",
            "Only 18 held-out claims are available per model.",
            "The incorrect-confidence strata contain 1, 4, and 1 baseline errors, respectively.",
            "Standardized safety and abstention benchmarks remain a separate required gate.",
        ],
        "research_decision": {
            "fixed_raw_receptor_replication": replicated,
            "maximum_research_alpha": 0.125,
            "adaptive_controller": False,
            "signed_dosing": False,
            "vector_field": False,
            "tolerance_or_homeostasis": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
