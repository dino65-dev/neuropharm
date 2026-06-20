"""Combine R2.5 subspace and cross-domain transfer evidence."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("subspace_analysis", type=Path)
    parser.add_argument("transfer", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    subspace = json.loads(args.subspace_analysis.read_text(encoding="utf-8"))
    transfer = json.loads(args.transfer.read_text(encoding="utf-8"))
    output = args.output or args.transfer.with_name(
        args.transfer.stem + "_analysis.json"
    )

    full = transfer["full_receptor_transfer"]
    leave_out = transfer["leave_target_domain_out_transfer"]
    matrix = transfer["domain_to_domain_transfer_matrix"]
    matrix_values = [
        result["antisymmetric_effect"]
        for targets in matrix.values()
        for result in targets.values()
    ]
    analysis = {
        "subspace_source": str(args.subspace_analysis),
        "transfer_source": str(args.transfer),
        "mean_receptor_transfer": {
            "test_domains": len(full),
            "positive_domains": sum(
                row["antisymmetric_effect"] > 0 for row in full.values()
            ),
            "mean_effect": statistics.fmean(
                row["antisymmetric_effect"] for row in full.values()
            ),
            "minimum_effect": min(
                row["antisymmetric_effect"] for row in full.values()
            ),
        },
        "leave_target_domain_out_transfer": {
            "test_domains": len(leave_out),
            "positive_domains": sum(
                row["antisymmetric_effect"] > 0 for row in leave_out.values()
            ),
            "mean_effect": statistics.fmean(
                row["antisymmetric_effect"] for row in leave_out.values()
            ),
            "minimum_effect": min(
                row["antisymmetric_effect"] for row in leave_out.values()
            ),
        },
        "domain_to_domain_matrix": {
            "source_domains": len(matrix),
            "target_domains": len(next(iter(matrix.values()))),
            "cells": len(matrix_values),
            "positive_cells": sum(value > 0 for value in matrix_values),
            "mean_effect": statistics.fmean(matrix_values),
            "minimum_effect": min(matrix_values),
            "maximum_effect": max(matrix_values),
        },
        "subspace_verdict": subspace["verdict"],
        "combined_verdict": (
            "The mean epistemic-assertiveness receptor is causally transferable "
            "across the tested domains. The centered variation is not yet a "
            "validated semantic receptor family: only four dimensions are "
            "principal-angle stable, and those axes are dominated by response-"
            "template wording. Preserve the mean receptor baseline and redesign "
            "the construction matrix as a full claim×question×response crossing "
            "before R3 subspace ligand synthesis."
        ),
        "limitations": [
            "Cross-domain tests reuse the same linguistic-certainty endpoint templates.",
            "Several source or target domains contain only one construction or test claim.",
            "The transfer matrix establishes sign consistency, not unique receptor identification.",
        ],
        "next_action": (
            "Create a fully crossed construction design, then repeat R2.5. Do "
            "not optimize a ligand over the current centered PCs."
        ),
        "cuda": transfer["cuda"],
        "elapsed_seconds": transfer["elapsed_seconds"],
    }
    output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

