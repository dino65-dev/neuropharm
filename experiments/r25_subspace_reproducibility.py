"""R2.5: centered receptor-subspace reproducibility.

Consumes the corrected R2 contrast matrix. No language-model forward passes
are required.

Measures:
  1. claim-grouped held-out reconstruction for k in {1,2,4,6,8,12,16};
  2. random-subspace and coordinate-permuted covariance nulls;
  3. principal-angle stability across independent claim halves;
  4. descriptive variance attribution of centered PC coordinates.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


K_VALUES = (1, 2, 4, 6, 8, 12, 16)


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    n = len(values)
    return {
        "mean": statistics.fmean(values),
        "standard_error": statistics.stdev(values) / math.sqrt(n),
        "median": statistics.median(values),
        "p05": ordered[int(0.05 * n)],
        "p95": ordered[min(n - 1, math.ceil(0.95 * n) - 1)],
        "minimum": min(values),
        "maximum": max(values),
    }


def question_template(prompt: str) -> str:
    if prompt.startswith("Assess whether"):
        return "assess"
    if prompt.startswith("What evidence supports"):
        return "evidence"
    if prompt.startswith("Give a concise judgment"):
        return "judgment"
    if prompt.startswith("Evaluate the following"):
        return "evaluate"
    raise ValueError(f"unknown question template: {prompt!r}")


def grouped_indices(metadata: list[dict[str, Any]]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[int(row["claim_id"])].append(index)
    return dict(groups)


def fit_centered_basis(
    values: torch.Tensor,
    max_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = values.mean(dim=0)
    centered = values - mean
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    return mean, vh[:max_k].T.contiguous()


def heldout_r2(
    heldout: torch.Tensor,
    train_mean: torch.Tensor,
    basis: torch.Tensor,
) -> float:
    centered = heldout - train_mean
    projected = centered @ basis @ basis.T
    residual_ss = (centered - projected).square().sum()
    total_ss = centered.square().sum().clamp_min(1e-30)
    return float((1 - residual_ss / total_ss).item())


def random_basis(
    dimension: int,
    k: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    matrix = torch.randn(
        dimension,
        k,
        generator=generator,
        dtype=torch.float32,
        device="cpu",
    ).to(device)
    q, _ = torch.linalg.qr(matrix, mode="reduced")
    return q


def coordinate_permuted(values: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    # Independently permuting each coordinate preserves every marginal
    # coordinate distribution while destroying cross-coordinate covariance.
    rows, dimensions = values.shape
    permutations = torch.stack(
        [torch.randperm(rows, generator=generator) for _ in range(dimensions)],
        dim=1,
    ).to(values.device)
    coordinate_indices = torch.arange(dimensions, device=values.device).unsqueeze(0)
    return values[permutations, coordinate_indices]


def marginal_eta_squared(
    coordinate: torch.Tensor,
    labels: list[str],
) -> float:
    values = coordinate.to(torch.float64).cpu()
    grand_mean = values.mean()
    total = (values - grand_mean).square().sum()
    if total <= 0:
        return 0.0
    groups: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    between = torch.tensor(0.0, dtype=torch.float64)
    for indices in groups.values():
        group = values[indices]
        between += len(indices) * (group.mean() - grand_mean).square()
    return float((between / total).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("contrast_matrix", type=Path)
    parser.add_argument("--replicates", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260620)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/pkpd/r25_subspace_reproducibility.json"),
    )
    parser.add_argument(
        "--basis-output",
        type=Path,
        default=Path("artifacts/pkpd/r25_validated_subspace.pt"),
    )
    args = parser.parse_args()

    start = time.time()
    bundle = torch.load(args.contrast_matrix, map_location="cpu", weights_only=False)
    differences = bundle["differences"].to(torch.float32)
    metadata = bundle["metadata"]
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    differences = differences.to(device)
    groups = grouped_indices(metadata)
    claim_ids = sorted(groups)
    if len(claim_ids) % 2:
        raise ValueError("claim-group halves require an even number of claims")
    if any(len(indices) != 4 for indices in groups.values()):
        raise ValueError("each claim must contribute exactly four question templates")

    torch_generator = torch.Generator(device="cpu").manual_seed(args.seed)
    py_rng = random.Random(args.seed)
    reconstruction = {
        str(k): {"observed": [], "random_subspace": [], "coordinate_permuted": []}
        for k in K_VALUES
    }
    angle_stability = {
        str(k): {"mean_cos2": [], "minimum_cosine": [], "cosines": []}
        for k in K_VALUES
    }

    for replicate in range(args.replicates):
        shuffled = claim_ids.copy()
        py_rng.shuffle(shuffled)
        midpoint = len(shuffled) // 2
        left_claims = shuffled[:midpoint]
        right_claims = shuffled[midpoint:]
        left_indices = [index for claim_id in left_claims for index in groups[claim_id]]
        right_indices = [index for claim_id in right_claims for index in groups[claim_id]]
        left = differences[left_indices]
        right = differences[right_indices]
        left_mean, left_basis = fit_centered_basis(left, max(K_VALUES))
        right_mean, right_basis = fit_centered_basis(right, max(K_VALUES))

        permuted_left = coordinate_permuted(left, torch_generator)
        permuted_mean, permuted_basis = fit_centered_basis(
            permuted_left, max(K_VALUES)
        )
        for k in K_VALUES:
            reconstruction[str(k)]["observed"].append(
                heldout_r2(right, left_mean, left_basis[:, :k])
            )
            random_subspace_basis = random_basis(
                differences.shape[1], k, device, torch_generator
            )
            reconstruction[str(k)]["random_subspace"].append(
                heldout_r2(right, left_mean, random_subspace_basis)
            )
            reconstruction[str(k)]["coordinate_permuted"].append(
                heldout_r2(right, permuted_mean, permuted_basis[:, :k])
            )

            singular_values = torch.linalg.svdvals(
                left_basis[:, :k].T @ right_basis[:, :k]
            ).clamp(0, 1)
            angle_stability[str(k)]["mean_cos2"].append(
                float(singular_values.square().mean().item())
            )
            angle_stability[str(k)]["minimum_cosine"].append(
                float(singular_values.min().item())
            )
            angle_stability[str(k)]["cosines"].append(
                [float(value) for value in singular_values.cpu()]
            )

        if (replicate + 1) % 50 == 0:
            print(f"replicate {replicate + 1}/{args.replicates}")

    reconstruction_summary: dict[str, Any] = {}
    for k in K_VALUES:
        key = str(k)
        observed = reconstruction[key]["observed"]
        random_null = reconstruction[key]["random_subspace"]
        permuted_null = reconstruction[key]["coordinate_permuted"]
        observed_minus_random = [
            value - null for value, null in zip(observed, random_null)
        ]
        observed_minus_permuted = [
            value - null for value, null in zip(observed, permuted_null)
        ]
        reconstruction_summary[key] = {
            "observed": summarize(observed),
            "random_subspace": summarize(random_null),
            "coordinate_permuted": summarize(permuted_null),
            "observed_minus_random": summarize(observed_minus_random),
            "observed_minus_permuted": summarize(observed_minus_permuted),
            "above_both_nulls": (
                summarize(observed_minus_random)["p05"] > 0
                and summarize(observed_minus_permuted)["p05"] > 0
            ),
        }

    eligible = [
        k for k in K_VALUES
        if reconstruction_summary[str(k)]["above_both_nulls"]
    ]
    if not eligible:
        selected_k = None
    else:
        best_k = max(
            eligible,
            key=lambda k: reconstruction_summary[str(k)]["observed"]["mean"],
        )
        best = reconstruction_summary[str(best_k)]["observed"]
        threshold = best["mean"] - best["standard_error"]
        selected_k = min(
            k for k in eligible
            if reconstruction_summary[str(k)]["observed"]["mean"] >= threshold
        )

    angle_summary = {
        str(k): {
            "mean_cos2": summarize(angle_stability[str(k)]["mean_cos2"]),
            "minimum_cosine": summarize(
                angle_stability[str(k)]["minimum_cosine"]
            ),
        }
        for k in K_VALUES
    }

    full_mean, full_basis = fit_centered_basis(differences, max(K_VALUES))
    centered = differences - full_mean
    coordinates = centered @ full_basis
    labels = {
        "claim": [str(row["claim_id"]) for row in metadata],
        "domain": [str(row["domain"]) for row in metadata],
        "question_template": [question_template(str(row["prompt"])) for row in metadata],
        "response_template": [str(row["response_pair_index"]) for row in metadata],
    }
    variance_attribution: dict[str, Any] = {}
    for axis in range(max(K_VALUES)):
        variance_attribution[str(axis + 1)] = {
            factor: marginal_eta_squared(coordinates[:, axis], factor_labels)
            for factor, factor_labels in labels.items()
        }

    payload = {
        "experiment": "R2.5_receptor_subspace_reproducibility",
        "source_contrast_matrix": str(args.contrast_matrix),
        "device": str(device),
        "n_claims": len(claim_ids),
        "n_contrasts": int(differences.shape[0]),
        "activation_dimension": int(differences.shape[1]),
        "replicates": args.replicates,
        "k_values": list(K_VALUES),
        "cross_validated_reconstruction": reconstruction_summary,
        "selection_rule": (
            "smallest k above both nulls and within one standard error of "
            "the best held-out reconstruction"
        ),
        "selected_k": selected_k,
        "principal_angle_stability": angle_summary,
        "variance_attribution": {
            "method": (
                "marginal one-way eta-squared per centered PC; factors are "
                "correlated, so fractions are descriptive and non-additive"
            ),
            "axes": variance_attribution,
        },
        "elapsed_seconds": time.time() - start,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.basis_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "mean": full_mean.cpu(),
        "centered_basis": full_basis.cpu(),
        "selected_k": selected_k,
        "k_values": list(K_VALUES),
        "layer": bundle["layer"],
        "site": bundle["site"],
        "claim_group_split": bundle["claim_group_split"],
    }, args.basis_output)
    print(args.output)


if __name__ == "__main__":
    main()
