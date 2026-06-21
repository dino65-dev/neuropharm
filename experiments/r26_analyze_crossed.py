"""R2.6 factorial semantic-nuisance decomposition and subspace gates."""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch

from experiments.r26_design import DOMAINS


K_VALUES = (1, 2, 4, 6, 8, 12, 16)


def fit_basis(values: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    mean = values.mean(dim=0)
    _, _, vh = torch.linalg.svd(values - mean, full_matrices=False)
    return mean, vh[: min(k, vh.shape[0])].T.contiguous()


def heldout_r2(
    values: torch.Tensor, train_mean: torch.Tensor, basis: torch.Tensor
) -> float:
    centered = values - train_mean
    reconstruction = (centered @ basis) @ basis.T
    residual_ss = (centered - reconstruction).square().sum()
    total_ss = centered.square().sum().clamp_min(1e-30)
    return float((1.0 - residual_ss / total_ss).item())


def principal_stability(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
    singular = torch.linalg.svdvals(left.T @ right).clamp(0, 1)
    return {
        "mean_cosine_squared": float(singular.square().mean().item()),
        "minimum_cosine": float(singular.min().item()),
    }


def random_basis(
    dimension: int, k: int, generator: torch.Generator, device: torch.device
) -> torch.Tensor:
    matrix = torch.randn(
        dimension, k, generator=generator, device="cpu", dtype=torch.float32
    ).to(device)
    return torch.linalg.qr(matrix, mode="reduced").Q


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    n = len(values)
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / math.sqrt(n) if n > 1 else 0.0
    return {
        "mean": mean,
        "standard_error": se,
        "lower_95": mean - 1.96 * se,
        "p05": ordered[max(0, math.ceil(0.05 * n) - 1)],
        "p95": ordered[min(n - 1, math.ceil(0.95 * n) - 1)],
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def effect_partition(contrasts: torch.Tensor, domains: list[str]) -> dict[str, Any]:
    """Orthogonally partition balanced certainty contrasts d[c,q,r]."""
    c_count, q_count, r_count, _ = contrasts.shape
    main = contrasts.mean(dim=(0, 1, 2))
    claim = contrasts.mean(dim=(1, 2)) - main
    question = contrasts.mean(dim=(0, 2)) - main
    response = contrasts.mean(dim=(0, 1)) - main
    residual = (
        contrasts
        - main
        - claim[:, None, None, :]
        - question[None, :, None, :]
        - response[None, None, :, :]
    )
    components = {
        "certainty_main": c_count * q_count * r_count * main.square().sum(),
        "certainty_x_claim": q_count * r_count * claim.square().sum(),
        "certainty_x_question": c_count * r_count * question.square().sum(),
        "certainty_x_response": c_count * q_count * response.square().sum(),
        "higher_interactions_and_residual": residual.square().sum(),
    }
    total = sum(components.values()).clamp_min(1e-30)

    domain_names = sorted(set(domains))
    domain_effects = torch.stack([
        claim[[index for index, value in enumerate(domains) if value == domain]].mean(0)
        for domain in domain_names
    ])
    domain_by_claim = torch.stack([
        domain_effects[domain_names.index(domain)] for domain in domains
    ])
    domain_ss = q_count * r_count * domain_by_claim.square().sum()
    within_domain_claim_ss = (
        q_count * r_count * (claim - domain_by_claim).square().sum()
    )
    return {
        "sum_of_squares": {
            name: float(value.item()) for name, value in components.items()
        },
        "fraction_of_contrast_energy": {
            name: float((value / total).item()) for name, value in components.items()
        },
        "claim_partition": {
            "certainty_x_domain_sum_of_squares": float(domain_ss.item()),
            "certainty_x_within_domain_claim_sum_of_squares": float(
                within_domain_claim_ss.item()
            ),
        },
    }


def wording_nuisance_basis(
    endpoints: torch.Tensor, energy_threshold: float
) -> tuple[torch.Tensor, dict[str, Any]]:
    # Average over claim and question before contrasting response frames at a
    # fixed certainty. These are same-certainty wording-only negative controls.
    wording_means = endpoints.mean(dim=(0, 1))  # [response, certainty, activation]
    controls = (
        wording_means - wording_means.mean(dim=0, keepdim=True)
    ).reshape(-1, endpoints.shape[-1])
    controls = controls[controls.norm(dim=1) > 1e-8]
    _, singular, vh = torch.linalg.svd(controls, full_matrices=False)
    energy = singular.square()
    cumulative = torch.cumsum(energy / energy.sum().clamp_min(1e-30), dim=0)
    rank = int(
        torch.searchsorted(
            cumulative,
            torch.tensor(energy_threshold, device=cumulative.device),
        ).item()
    ) + 1
    basis = vh[:rank].T.contiguous()
    return basis, {
        "control_matrix_shape": list(controls.shape),
        "selected_rank": rank,
        "energy_threshold": energy_threshold,
        "captured_energy": float(cumulative[rank - 1].item()),
        "singular_values": [float(value) for value in singular.cpu()],
    }


def project_out(vector: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    residual = vector - basis @ (basis.T @ vector)
    return residual / residual.norm().clamp_min(1e-12)


def ridge_classifier_accuracy(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    classes: int,
    ridge: float = 1e-3,
) -> float:
    mean = train_x.mean(0, keepdim=True)
    scale = train_x.std(0, keepdim=True).clamp_min(1e-6)
    train = (train_x - mean) / scale
    test = (test_x - mean) / scale
    train = torch.cat((train, torch.ones(len(train), 1, device=train.device)), 1)
    test = torch.cat((test, torch.ones(len(test), 1, device=test.device)), 1)
    targets = torch.nn.functional.one_hot(train_y, classes).to(train.dtype)
    identity = torch.eye(train.shape[1], device=train.device, dtype=train.dtype)
    identity[-1, -1] = 0
    weights = torch.linalg.solve(
        train.T @ train + ridge * identity, train.T @ targets
    )
    predictions = (test @ weights).argmax(1)
    return float((predictions == test_y).float().mean().item())


def construction_claim_folds(
    construction_ids: list[int], claim_domains: list[str]
) -> list[list[int]]:
    by_domain = {
        domain: [claim_id for claim_id in construction_ids if claim_domains[claim_id] == domain]
        for domain in DOMAINS
    }
    if any(len(ids) != 4 for ids in by_domain.values()):
        raise ValueError("construction split must contain four claims per domain")
    return [[by_domain[domain][fold] for domain in DOMAINS] for fold in range(4)]


def family_splits(families: list[str], positions: list[str]) -> list[dict[str, Any]]:
    unique_families = sorted(set(families))
    splits: list[dict[str, Any]] = []
    # Three unique 2-vs-2 family partitions.
    for left in itertools.combinations(unique_families, 2):
        right = tuple(value for value in unique_families if value not in left)
        if left > right:
            continue
        splits.append({
            "name": f"families_{'+'.join(left)}__vs__{'+'.join(right)}",
            "left": [i for i, value in enumerate(families) if value in left],
            "right": [i for i, value in enumerate(families) if value in right],
        })
    # Four leave-one-family-out environments.
    for heldout in unique_families:
        splits.append({
            "name": f"leave_{heldout}_out",
            "left": [i for i, value in enumerate(families) if value != heldout],
            "right": [i for i, value in enumerate(families) if value == heldout],
        })
    splits.append({
        "name": "sentence_initial__vs__sentence_final",
        "left": [i for i, value in enumerate(positions) if value == "initial"],
        "right": [i for i, value in enumerate(positions) if value == "final"],
    })
    return splits


def one_se_selection(rows: dict[int, dict[str, Any]]) -> int:
    best_k = max(rows, key=lambda key: rows[key]["heldout_r2"]["mean"])
    threshold = (
        rows[best_k]["heldout_r2"]["mean"]
        - rows[best_k]["heldout_r2"]["standard_error"]
    )
    return min(k for k in rows if rows[k]["heldout_r2"]["mean"] >= threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("endpoints", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--nuisance-energy", type=float, default=0.99)
    parser.add_argument("--random-stability-replicates", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/pkpd/r26_crossed_analysis_t4.json"),
    )
    parser.add_argument(
        "--basis-output",
        type=Path,
        default=Path("artifacts/pkpd/r26_semantic_basis_t4.pt"),
    )
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    bundle = torch.load(args.endpoints, map_location="cpu", weights_only=False)
    endpoints = bundle["endpoints"].to(device=device, dtype=torch.float32)
    if tuple(endpoints.shape[:4]) != (72, 4, 8, 2):
        raise ValueError(f"unexpected endpoint tensor shape {tuple(endpoints.shape)}")
    contrasts = endpoints[..., 1, :] - endpoints[..., 0, :]
    split = bundle["claim_split"]
    construction = [int(value) for value in split["construction"]]
    validation = [int(value) for value in split["validation"]]
    test = [int(value) for value in split["test"]]
    claim_domains = [str(row["domain"]) for row in bundle["claims"]]
    frame_families = [str(row["family"]) for row in bundle["response_frames"]]
    frame_positions = [str(row["position"]) for row in bundle["response_frames"]]

    construction_contrasts = contrasts[construction]
    receptor = construction_contrasts.mean(dim=(0, 1, 2))
    receptor = receptor / receptor.norm().clamp_min(1e-12)
    nuisance_basis, nuisance_report = wording_nuisance_basis(
        endpoints[construction], args.nuisance_energy
    )
    nuisance_fraction = float((nuisance_basis.T @ receptor).square().sum().item())
    receptor_perp = project_out(receptor, nuisance_basis)

    splits = family_splits(frame_families, frame_positions)
    folds = construction_claim_folds(construction, claim_domains)
    nested_rows: dict[int, dict[str, Any]] = {}
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    for k in K_VALUES:
        heldout_scores: list[float] = []
        null_scores: list[float] = []
        for environment in splits:
            for heldout_claims in folds:
                train_claims = [
                    claim_id for claim_id in construction
                    if claim_id not in heldout_claims
                ]
                train_values = contrasts[train_claims][
                    :, :, environment["left"], :
                ].mean(2).reshape(-1, contrasts.shape[-1])
                test_values = contrasts[heldout_claims][
                    :, :, environment["right"], :
                ].mean(2).reshape(-1, contrasts.shape[-1])
                mean, basis = fit_basis(train_values, k)
                heldout_scores.append(heldout_r2(test_values, mean, basis))
                null = random_basis(contrasts.shape[-1], k, generator, device)
                null_scores.append(heldout_r2(test_values, mean, null))
        nested_rows[k] = {
            "heldout_r2": summarize(heldout_scores),
            "matched_random_r2": summarize(null_scores),
            "folds": len(heldout_scores),
        }
    selected_k = one_se_selection(nested_rows)

    stability_by_k: dict[int, dict[str, Any]] = {}
    final_reconstruction_by_k: dict[int, dict[str, Any]] = {}
    for k in K_VALUES:
        stability_scores = []
        minimum_cosines = []
        reconstruction_scores = []
        reconstruction_null = []
        for environment in splits:
            left_values = contrasts[construction][
                :, :, environment["left"], :
            ].mean(2).reshape(-1, contrasts.shape[-1])
            right_values = contrasts[construction][
                :, :, environment["right"], :
            ].mean(2).reshape(-1, contrasts.shape[-1])
            left_mean, left_basis = fit_basis(left_values, k)
            _, right_basis = fit_basis(right_values, k)
            stability = principal_stability(left_basis, right_basis)
            stability_scores.append(stability["mean_cosine_squared"])
            minimum_cosines.append(stability["minimum_cosine"])
            validation_values = contrasts[validation][
                :, :, environment["right"], :
            ].mean(2).reshape(-1, contrasts.shape[-1])
            reconstruction_scores.append(
                heldout_r2(validation_values, left_mean, left_basis)
            )
            null = random_basis(contrasts.shape[-1], k, generator, device)
            reconstruction_null.append(
                heldout_r2(validation_values, left_mean, null)
            )
        random_stability = []
        for _ in range(args.random_stability_replicates):
            random_stability.append(principal_stability(
                random_basis(contrasts.shape[-1], k, generator, device),
                random_basis(contrasts.shape[-1], k, generator, device),
            )["mean_cosine_squared"])
        stability_by_k[k] = {
            "mean_cosine_squared": summarize(stability_scores),
            "minimum_cosine": summarize(minimum_cosines),
            "matched_random_mean_cosine_squared": summarize(random_stability),
        }
        final_reconstruction_by_k[k] = {
            "heldout_response_family_r2": summarize(reconstruction_scores),
            "matched_random_r2": summarize(reconstruction_null),
        }

    # Fit the candidate semantic coordinates only after averaging every response
    # frame equally within each construction claim-question cell.
    averaged = contrasts.mean(dim=2)
    semantic_mean, semantic_basis = fit_basis(
        averaged[construction].reshape(-1, contrasts.shape[-1]), selected_k
    )

    def coordinate_rows(claim_ids: list[int]):
        values = contrasts[claim_ids] - semantic_mean
        coordinates = values @ semantic_basis
        count = len(claim_ids) * contrasts.shape[1]
        family_labels = torch.tensor(
            [frame_families.index(frame_families[r]) for _ in range(count) for r in range(8)],
            device=device,
        )
        # Map repeated family strings to compact class IDs.
        family_map = {value: index for index, value in enumerate(sorted(set(frame_families)))}
        family_labels = torch.tensor(
            [family_map[frame_families[r]] for _ in range(count) for r in range(8)],
            device=device,
        )
        frame_labels = torch.tensor(
            [r for _ in range(count) for r in range(8)], device=device
        )
        position_map = {"initial": 0, "final": 1}
        position_labels = torch.tensor(
            [position_map[frame_positions[r]] for _ in range(count) for r in range(8)],
            device=device,
        )
        return (
            coordinates.reshape(-1, selected_k),
            family_labels,
            frame_labels,
            position_labels,
        )

    train_coordinates = coordinate_rows(construction)
    validation_coordinates = coordinate_rows(validation)
    leakage = {
        "response_family_accuracy": ridge_classifier_accuracy(
            train_coordinates[0], train_coordinates[1],
            validation_coordinates[0], validation_coordinates[1], 4,
        ),
        "response_family_chance": 0.25,
        "response_frame_accuracy": ridge_classifier_accuracy(
            train_coordinates[0], train_coordinates[2],
            validation_coordinates[0], validation_coordinates[2], 8,
        ),
        "response_frame_chance": 0.125,
        "phrase_position_accuracy": ridge_classifier_accuracy(
            train_coordinates[0], train_coordinates[3],
            validation_coordinates[0], validation_coordinates[3], 2,
        ),
        "phrase_position_chance": 0.5,
    }

    def certainty_features(claim_ids: list[int], directions: torch.Tensor):
        values = endpoints[claim_ids]
        features = values @ directions
        labels = torch.arange(2, device=device).view(1, 1, 1, 2).expand(
            len(claim_ids), 4, 8, 2
        )
        return features.reshape(-1, directions.shape[1]), labels.reshape(-1)

    raw_directions = torch.cat((receptor[:, None], semantic_basis), dim=1)
    cleaned_directions = torch.cat((receptor_perp[:, None], semantic_basis), dim=1)
    raw_train = certainty_features(construction, raw_directions)
    raw_validation = certainty_features(validation, raw_directions)
    clean_train = certainty_features(construction, cleaned_directions)
    clean_validation = certainty_features(validation, cleaned_directions)
    certainty_containment = {
        "raw_semantic_coordinates_accuracy": ridge_classifier_accuracy(
            raw_train[0], raw_train[1], raw_validation[0], raw_validation[1], 2
        ),
        "nuisance_projected_coordinates_accuracy": ridge_classifier_accuracy(
            clean_train[0], clean_train[1],
            clean_validation[0], clean_validation[1], 2,
        ),
        "chance": 0.5,
        "validation_pairwise_positive_fraction_r0": float(
            ((contrasts[validation] @ receptor) > 0).float().mean().item()
        ),
        "validation_pairwise_positive_fraction_r_perp_nuisance": float(
            ((contrasts[validation] @ receptor_perp) > 0).float().mean().item()
        ),
        "test_pairwise_positive_fraction_r0": float(
            ((contrasts[test] @ receptor) > 0).float().mean().item()
        ),
        "test_pairwise_positive_fraction_r_perp_nuisance": float(
            ((contrasts[test] @ receptor_perp) > 0).float().mean().item()
        ),
    }

    selected_stability = stability_by_k[selected_k]
    selected_reconstruction = final_reconstruction_by_k[selected_k]
    gates = {
        "template_disjoint_stability": (
            selected_stability["mean_cosine_squared"]["lower_95"]
            > selected_stability["matched_random_mean_cosine_squared"]["p95"]
        ),
        "heldout_response_reconstruction": (
            selected_reconstruction["heldout_response_family_r2"]["lower_95"]
            > selected_reconstruction["matched_random_r2"]["p95"]
        ),
        "low_response_family_leakage": (
            leakage["response_family_accuracy"] <= 0.35
        ),
        "preserved_certainty_information": (
            certainty_containment["nuisance_projected_coordinates_accuracy"] >= 0.70
        ),
    }
    payload = {
        "experiment": "R2.6_fully_crossed_semantic_nuisance_decomposition",
        "source": str(args.endpoints),
        "tensor_shape": list(endpoints.shape),
        "claim_split": split,
        "factorial_certainty_effect_partition": effect_partition(
            construction_contrasts,
            [claim_domains[index] for index in construction],
        ),
        "mean_receptor": {
            "norm_before_normalization": float(
                construction_contrasts.mean(dim=(0, 1, 2)).norm().item()
            ),
            "wording_nuisance_projection_fraction": nuisance_fraction,
            "cosine_raw_vs_projected": float(
                torch.dot(receptor, receptor_perp).item()
            ),
            "wording_nuisance": nuisance_report,
        },
        "nested_claim_and_response_environment_cv": {
            "candidate_k": list(K_VALUES),
            "environment_count": len(splits),
            "claim_folds_per_environment": len(folds),
            "rows": {str(k): value for k, value in nested_rows.items()},
            "one_standard_error_selected_k": selected_k,
        },
        "template_disjoint_stability": {
            str(k): value for k, value in stability_by_k.items()
        },
        "heldout_response_family_reconstruction": {
            str(k): value for k, value in final_reconstruction_by_k.items()
        },
        "template_leakage": leakage,
        "certainty_containment": certainty_containment,
        "pass_gates": gates,
        "all_noncausal_gates_passed": all(gates.values()),
        "limitations": [
            "The wording nuisance basis estimates stable response-frame effects after averaging claims and questions.",
            "Linear leakage and certainty classifiers are claim-disjoint but use repeated question and response cells within each held-out claim.",
            "Causal unseen-family transfer is evaluated in a separate GPU experiment.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.basis_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "source": str(args.endpoints),
        "selected_k": selected_k,
        "mean_receptor": receptor.detach().cpu(),
        "wording_nuisance_basis": nuisance_basis.detach().cpu(),
        "nuisance_projected_receptor": receptor_perp.detach().cpu(),
        "semantic_mean": semantic_mean.detach().cpu(),
        "semantic_basis": semantic_basis.detach().cpu(),
        "pass_gates": gates,
    }, args.basis_output)
    print(args.output)
    print(args.basis_output)


if __name__ == "__main__":
    main()
