from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from pharmacokinetics.state_space import (
    NeuropharmacologyModel,
    PKPDState,
    PKPDStep,
)


DoseSchedule = Callable[[int], torch.Tensor]


def protocol_schedule(
    protocol: str,
    dose: float,
    n_compounds: int = 1,
    stop: int = 40,
    pulse_period: int = 8,
) -> DoseSchedule:
    """Create a deterministic token-step dose schedule."""
    if protocol not in {"zero", "bolus", "infusion", "pulses"}:
        raise ValueError(f"unknown protocol: {protocol}")

    def schedule(step_index: int) -> torch.Tensor:
        active = {
            "zero": False,
            "bolus": step_index == 0,
            "infusion": step_index < stop,
            "pulses": step_index < stop and step_index % pulse_period == 0,
        }[protocol]
        value = float(dose) if active else 0.0
        return torch.full((n_compounds,), value, dtype=torch.float64)

    return schedule


@dataclass
class PKPDGenerationController:
    """Stateful batch-size-one hook advancing once per forward call.

    The prompt prefill is one PK/PD step even when ``resid.shape[1] > 1``.
    Every later cached decode call is also one step. Only position ``-1`` is
    modified, so prompt length never changes the pharmacological clock.
    """

    model: NeuropharmacologyModel
    dose_schedule: DoseSchedule
    state: PKPDState | None = None
    step_index: int = 0
    trace: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.state is None:
            self.state = self.model.initial_state()

    def reset(self) -> None:
        self.state = self.model.initial_state()
        self.step_index = 0
        self.trace.clear()

    def advance(self) -> PKPDStep:
        assert self.state is not None
        dose = self.dose_schedule(self.step_index).to(
            device=self.state.concentration.device,
            dtype=self.state.concentration.dtype,
        )
        state_before = self.state
        step = self.model.step(state_before, dose)
        self.state = step.state
        self.trace.append({
            "pkpd_step_index": self.step_index,
            "dose": dose.detach().to(torch.float64).cpu().tolist(),
            "concentration": step.state.concentration.detach().to(torch.float64).cpu().tolist(),
            "occupancy": step.occupancy.detach().to(torch.float64).cpu().tolist(),
            "sensitivity": step.sensitivity_used.detach().to(torch.float64).cpu().tolist(),
            "adaptation": step.adaptation_used.detach().to(torch.float64).cpu().tolist(),
            "sensitivity_next": step.state.sensitivity.detach().to(torch.float64).cpu().tolist(),
            "adaptation_next": step.state.adaptation.detach().to(torch.float64).cpu().tolist(),
            "receptor_effect": step.receptor_effect.detach().to(torch.float64).cpu().tolist(),
            "delta_h_norm": float(step.delta_h.to(torch.float64).norm().item()),
        })
        self.step_index += 1
        return step

    def hook(self, resid: torch.Tensor, hook_point: object | None = None) -> torch.Tensor:
        if resid.ndim != 3:
            raise ValueError(
                f"resid must have shape (batch, position, d_model), got {tuple(resid.shape)}"
            )
        if resid.shape[0] != 1:
            raise ValueError(f"initial controller requires batch size 1, got {resid.shape[0]}")
        if resid.shape[-1] != self.model.d_model:
            raise ValueError(
                f"residual width must be {self.model.d_model}, got {resid.shape[-1]}"
            )

        step = self.advance()
        self.trace[-1]["forward_sequence_length"] = int(resid.shape[1])
        self.trace[-1]["is_prefill"] = bool(resid.shape[1] > 1 and self.step_index == 1)
        self.trace[-1]["injected_token_position"] = int(resid.shape[1] - 1)

        delta = step.delta_h.to(device=resid.device, dtype=resid.dtype)
        if torch.count_nonzero(delta).item() == 0:
            return resid
        result = resid.clone()
        result[:, -1, :] = result[:, -1, :] + delta
        return result

    def record_outcome(
        self,
        *,
        next_token_id: int,
        next_token: str,
        target_probe_score: float,
        kl_from_baseline: float,
    ) -> None:
        if not self.trace:
            raise RuntimeError("cannot record an outcome before a PK/PD step")
        self.trace[-1].update({
            "next_token_id": int(next_token_id),
            "next_token": next_token,
            "target_probe_score": float(target_probe_score),
            "kl_from_baseline": float(kl_from_baseline),
        })

