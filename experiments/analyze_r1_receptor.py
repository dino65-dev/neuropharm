"""Add prompt-bootstrap uncertainty and a strict verdict to an R1 artifact."""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path


def bootstrap_ci(
    values: list[float],
    seed: int,
    samples: int = 20_000,
) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(samples)
    )
    return [
        estimates[int(0.025 * samples)],
        estimates[int(0.975 * samples)],
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    output = args.output or args.input.with_name(args.input.stem + "_analysis.json")
    test = payload["test"]
    receptor = test["receptor_prompt_deltas"]
    opposite = test["opposite_prompt_deltas"]
    symmetry = [
        (positive - negative) / 2
        for positive, negative in zip(receptor, opposite)
    ]
    random_effects = test["random_effects"]

    analysis = {
        "source": str(args.input),
        "gpu_verified": {
            "device": payload["cuda_device"],
            "capability": payload["cuda_capability"],
            "torch_version": payload["torch_version"],
            "cuda_build": payload["cuda_build"],
            "peak_cuda_allocated_gib": (
                payload["peak_cuda_allocated_bytes"] / 2**30
            ),
            "peak_cuda_reserved_gib": (
                payload["peak_cuda_reserved_bytes"] / 2**30
            ),
        },
        "test": {
            "receptor_mean_delta": statistics.fmean(receptor),
            "receptor_95pct_prompt_bootstrap": bootstrap_ci(receptor, seed=1),
            "opposite_mean_delta": statistics.fmean(opposite),
            "opposite_95pct_prompt_bootstrap": bootstrap_ci(opposite, seed=2),
            "symmetry_mean": statistics.fmean(symmetry),
            "symmetry_95pct_prompt_bootstrap": bootstrap_ci(symmetry, seed=3),
            "random_count": len(random_effects),
            "random_mean": statistics.fmean(random_effects),
            "random_std": statistics.stdev(random_effects),
            "random_min": min(random_effects),
            "random_max": max(random_effects),
            "random_q95": test["random_q95"],
            "receptor_percentile": test["receptor_percentile"],
        },
        "gates": {
            "sign": statistics.fmean(receptor) > 0
            and statistics.fmean(opposite) < 0,
            "symmetry": bootstrap_ci(symmetry, seed=3)[0] > 0,
            "random_null": statistics.fmean(receptor) > test["random_q95"],
        },
    }
    analysis["overall_passed"] = all(analysis["gates"].values())
    analysis["decision"] = (
        "Proceed to receptor-candidate and layer/site comparison."
        if analysis["overall_passed"]
        else (
            "Do not enable tolerance or homeostasis. The receptor has a "
            "replicated signed effect but does not exceed the matched random "
            "direction null."
        )
    )
    output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

