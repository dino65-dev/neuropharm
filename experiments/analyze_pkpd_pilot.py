"""Audit a PK/PD Qwen pilot and apply prespecified pass/fail gates."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def bootstrap_ci(values: list[float], seed: int, samples: int = 10_000) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(
        sum(values[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(samples)
    )
    return [means[int(0.025 * samples)], means[int(0.975 * samples)]]


def prompt_mean(result: dict[str, Any], condition: str, key: str) -> float:
    trace = result["conditions"][condition]["trace"]
    return sum(row[key] for row in trace) / len(trace)


def audit_traces(payload: dict[str, Any]) -> list[list[Any]]:
    issues: list[list[Any]] = []
    for result in payload["results"]:
        for name, condition in result["conditions"].items():
            trace = condition["trace"]
            expected = list(range(len(trace)))
            observed = [row["pkpd_step_index"] for row in trace]
            if observed != expected:
                issues.append([result["prompt_index"], name, "step_indices"])
            if not trace[0]["is_prefill"] or trace[0]["forward_sequence_length"] <= 1:
                issues.append([result["prompt_index"], name, "prefill"])
            if any(row["forward_sequence_length"] != 1 for row in trace[1:]):
                issues.append([result["prompt_index"], name, "decode_shape"])
            for row in trace:
                values: list[float] = []
                for key in (
                    "concentration",
                    "occupancy",
                    "sensitivity",
                    "adaptation",
                    "receptor_effect",
                ):
                    value = row[key]
                    if isinstance(value, list):
                        values.extend(value)
                values.extend([
                    row["delta_h_norm"],
                    row["target_probe_score"],
                    row["kl_from_baseline"],
                ])
                if not all(math.isfinite(value) for value in values):
                    issues.append([result["prompt_index"], name, "nonfinite"])
                occupancy = row["occupancy"]
                if isinstance(occupancy, list) and any(
                    value < 0 or value > 1 for value in occupancy
                ):
                    issues.append([result["prompt_index"], name, "occupancy_bounds"])
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    output = args.output or args.input.with_name(args.input.stem + "_analysis.json")
    conditions = list(payload["results"][0]["conditions"])

    summary: dict[str, Any] = {}
    for condition in conditions:
        trajectory_delta = [
            prompt_mean(result, condition, "target_probe_score")
            - prompt_mean(result, "baseline", "target_probe_score")
            for result in payload["results"]
        ]
        first_token_delta = [
            result["conditions"][condition]["trace"][0]["target_probe_score"]
            - result["conditions"]["baseline"]["trace"][0]["target_probe_score"]
            for result in payload["results"]
        ]
        summary[condition] = {
            "mean_first_token_target_delta": sum(first_token_delta) / len(first_token_delta),
            "first_token_target_delta_95pct_paired_bootstrap": bootstrap_ci(
                first_token_delta, seed=0
            ),
            "mean_trajectory_target_delta_prefix_confounded": (
                sum(trajectory_delta) / len(trajectory_delta)
            ),
            "mean_first_token_kl": sum(
                result["conditions"][condition]["trace"][0]["kl_from_baseline"]
                for result in payload["results"]
            ) / len(payload["results"]),
            "mean_total_intervention": sum(
                result["conditions"][condition]["total_intervention"]
                for result in payload["results"]
            ) / len(payload["results"]),
            "generated_text_changed_prompts": sum(
                result["conditions"][condition]["text"]
                != result["conditions"]["baseline"]["text"]
                for result in payload["results"]
            ),
        }

    pulse_first = [
        result["conditions"]["pulses"]["trace"][0]["target_probe_score"]
        - result["conditions"]["baseline"]["trace"][0]["target_probe_score"]
        for result in payload["results"]
    ]
    random_first = [
        result["conditions"]["random_norm_matched"]["trace"][0]["target_probe_score"]
        - result["conditions"]["baseline"]["trace"][0]["target_probe_score"]
        for result in payload["results"]
    ]
    pulse_minus_random = [
        pulse - random for pulse, random in zip(pulse_first, random_first)
    ]
    pulse_minus_random_ci = bootstrap_ci(pulse_minus_random, seed=1)
    trace_issues = audit_traces(payload)
    matched_difference = abs(
        summary["static_matched"]["mean_total_intervention"]
        - summary["pulses"]["mean_total_intervention"]
    )
    causal_specificity_passed = (
        len(payload["results"]) >= 20 and pulse_minus_random_ci[0] > 0
    )
    pulse_counts = [
        sum(
            any(value > 0 for value in row["dose"])
            for row in result["conditions"]["pulses"]["trace"]
        )
        for result in payload["results"]
    ]
    repeated_pulse_exposure_passed = min(pulse_counts) >= 2

    analysis = {
        "source": str(args.input),
        "gates": {
            "trace_semantics": {"passed": not trace_issues, "issues": trace_issues},
            "zero_dose_identity": {
                "passed": payload["max_zero_dose_logit_error"] < 1e-6,
                "max_logit_error": payload["max_zero_dose_logit_error"],
            },
            "finite_bounded_occupancy": {"passed": not trace_issues},
            "static_pulse_total_intervention_match": {
                "passed": matched_difference < 1e-6,
                "absolute_difference": matched_difference,
            },
            "repeated_pulse_exposure": {
                "passed": repeated_pulse_exposure_passed,
                "pulse_counts_per_prompt": pulse_counts,
            },
            "causal_specificity_first_token": {
                "passed": causal_specificity_passed,
                "minimum_prompt_count": 20,
                "observed_prompt_count": len(payload["results"]),
                "pulse_minus_random_mean": (
                    sum(pulse_minus_random) / len(pulse_minus_random)
                ),
                "pulse_minus_random_95pct_paired_bootstrap": pulse_minus_random_ci,
            },
        },
        "overall_behavioral_gate_passed": (
            causal_specificity_passed and repeated_pulse_exposure_passed
        ),
        "conditions": summary,
        "interpretation": (
            "The first-token comparison is the unconfounded causal score because "
            "all conditions share the same prefix. Full-trajectory score and KL "
            "are prefix-confounded after generated tokens diverge. The target "
            "metric is a lexical log-probability contrast, not a validated "
            "behavior classifier."
        ),
    }
    output.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
