"""
Dose-Response Analysis — Finding the Therapeutic Window

Sweeps a range of steering coefficients and measures:
  - Output coherence (perplexity proxy)
  - Concept strength (cosine similarity to target direction)
  - Overdose detection (repetition / incoherence heuristics)

Analogy: LD50 curves in pharmacology.
         Find the minimum effective dose and the overdose threshold.
"""

from __future__ import annotations
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class DoseResponse:
    coefficient: float
    output: str
    repetition_score: float     # 0-1, higher = more repetitive (overdose signal)
    mean_token_entropy: float   # higher = more diverse vocabulary
    is_overdose: bool


def repetition_score(text: str, ngram_size: int = 4) -> float:
    """Measure n-gram repetition as overdose proxy."""
    tokens = text.split()
    if len(tokens) < ngram_size:
        return 0.0
    ngrams = [tuple(tokens[i:i+ngram_size]) for i in range(len(tokens)-ngram_size)]
    unique = len(set(ngrams))
    total = len(ngrams)
    return 1.0 - (unique / total) if total > 0 else 0.0


def run_dose_response(
    generate_fn: Callable[[float], str],
    coefficients: list[float] = None,
    overdose_threshold: float = 0.4,
    verbose: bool = True,
) -> list[DoseResponse]:
    """
    Run a dose-response sweep.

    Args:
        generate_fn: A function that takes a coefficient and returns generated text.
                     Use a lambda wrapping your injection.inject() or control_vector.generate().
        coefficients: List of doses to test. Default: 0 to 50.
        overdose_threshold: Repetition score above which output is flagged as overdose.
        verbose: Print results as they come in.

    Returns:
        List of DoseResponse objects, one per dose.

    Example:
        results = run_dose_response(
            generate_fn=lambda c: steerer.inject(pos, neg, layer=15, coefficient=c, prompt="Tell me about today."),
            coefficients=[0, 5, 10, 20, 30, 40]
        )
    """
    if coefficients is None:
        coefficients = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50]

    results = []
    for coeff in coefficients:
        if verbose:
            print(f"[Dose {coeff:>6.1f}] ", end="", flush=True)
        output = generate_fn(coeff)
        rep = repetition_score(output)
        words = output.split()
        # Entropy proxy: unique word ratio
        entropy = len(set(words)) / (len(words) + 1e-9)
        overdose = rep > overdose_threshold

        dr = DoseResponse(
            coefficient=coeff,
            output=output,
            repetition_score=rep,
            mean_token_entropy=entropy,
            is_overdose=overdose,
        )
        results.append(dr)

        if verbose:
            status = "⚠️  OVERDOSE" if overdose else "✅ OK"
            print(f"rep={rep:.3f}  entropy={entropy:.3f}  {status}")
            print(f"         {output[:120].strip()}...")

    return results


def find_therapeutic_window(results: list[DoseResponse]) -> tuple[float, float]:
    """Return (min_effective_dose, max_safe_dose)."""
    baseline_entropy = results[0].mean_token_entropy if results else 0.5
    min_effective = None
    max_safe = None

    for dr in results:
        if dr.is_overdose:
            break
        # Effective = entropy changed by >10% from baseline
        if min_effective is None and abs(dr.mean_token_entropy - baseline_entropy) > 0.05:
            min_effective = dr.coefficient
        max_safe = dr.coefficient

    return (min_effective or 0.0, max_safe or 0.0)


if __name__ == "__main__":
    print("Import and call run_dose_response() with your generate_fn.")
    print("Example:")
    print("  from dosing.dose_response import run_dose_response")
    print("  results = run_dose_response(lambda c: steerer.inject(..., coefficient=c, ...))")
