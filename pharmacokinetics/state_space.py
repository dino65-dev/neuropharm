from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PKPDParameters:
    """Parameters for ``m`` compounds, ``k`` receptors, dimension ``d``.

    Shapes:
      receptor_basis: ``(d, k)``
      affinity: ``(k, m)``
      ec50, hill, emax, recovery, desensitization: ``(k,)``
      elimination, absorption: ``(m,)``
    """

    receptor_basis: torch.Tensor
    affinity: torch.Tensor
    ec50: torch.Tensor
    hill: torch.Tensor
    emax: torch.Tensor
    elimination: torch.Tensor
    absorption: torch.Tensor
    recovery: torch.Tensor
    desensitization: torch.Tensor
    homeostasis_decay: float = 0.95
    homeostasis_gain: float = 0.0

    def __post_init__(self) -> None:
        d, k = self.receptor_basis.shape
        if self.affinity.ndim != 2 or self.affinity.shape[0] != k:
            raise ValueError(f"affinity must have shape ({k}, m), got {tuple(self.affinity.shape)}")
        m = self.affinity.shape[1]
        for name in ("ec50", "hill", "emax", "recovery", "desensitization"):
            value = getattr(self, name)
            if value.shape != (k,):
                raise ValueError(f"{name} must have shape ({k},), got {tuple(value.shape)}")
        for name in ("elimination", "absorption"):
            value = getattr(self, name)
            if value.shape != (m,):
                raise ValueError(f"{name} must have shape ({m},), got {tuple(value.shape)}")
        if d < 1 or k < 1 or m < 1:
            raise ValueError("d, k, and m must be positive")
        if torch.any(self.affinity < 0):
            raise ValueError("affinity must be nonnegative; encode effect sign in receptor_basis or emax")
        if torch.any(self.ec50 <= 0) or torch.any(self.hill <= 0):
            raise ValueError("ec50 and hill coefficients must be positive")
        if torch.any((self.elimination < 0) | (self.elimination >= 1)):
            raise ValueError("elimination must be in [0, 1)")
        if torch.any((self.absorption <= 0) | (self.absorption > 1)):
            raise ValueError("absorption must be in (0, 1]")
        if not 0 <= self.homeostasis_decay < 1:
            raise ValueError("homeostasis_decay must be in [0, 1)")


@dataclass
class PKPDState:
    concentration: torch.Tensor  # (m,)
    depot: torch.Tensor  # (m,)
    sensitivity: torch.Tensor  # (k,)
    adaptation: torch.Tensor  # (k,)

    def clone(self) -> "PKPDState":
        return PKPDState(
            concentration=self.concentration.clone(),
            depot=self.depot.clone(),
            sensitivity=self.sensitivity.clone(),
            adaptation=self.adaptation.clone(),
        )


@dataclass
class PKPDStep:
    state: PKPDState
    occupancy: torch.Tensor  # (k,)
    receptor_effect: torch.Tensor  # (k,)
    delta_h: torch.Tensor  # (d,)


class NeuropharmacologyModel:
    """Discrete token-level PK/PD state-space model."""

    def __init__(self, parameters: PKPDParameters):
        self.p = parameters
        self.d_model, self.n_receptors = parameters.receptor_basis.shape
        self.n_compounds = parameters.affinity.shape[1]

    def initial_state(self) -> PKPDState:
        ref = self.p.receptor_basis
        return PKPDState(
            concentration=torch.zeros(self.n_compounds, device=ref.device, dtype=ref.dtype),
            depot=torch.zeros(self.n_compounds, device=ref.device, dtype=ref.dtype),
            sensitivity=torch.ones(self.n_receptors, device=ref.device, dtype=ref.dtype),
            adaptation=torch.zeros(self.n_receptors, device=ref.device, dtype=ref.dtype),
        )

    def step(
        self,
        state: PKPDState,
        dose: torch.Tensor,
        availability: torch.Tensor | None = None,
    ) -> PKPDStep:
        """Advance one token.

        ``dose`` has shape ``(m,)`` and ``availability`` has shape ``(k,)``.
        The returned residual intervention ``delta_h`` has shape ``(d,)``.
        """
        if dose.shape != (self.n_compounds,):
            raise ValueError(f"dose must have shape ({self.n_compounds},), got {tuple(dose.shape)}")
        if torch.any(dose < 0):
            raise ValueError("dose must be nonnegative")
        if availability is None:
            availability = torch.ones_like(state.sensitivity)
        if availability.shape != (self.n_receptors,):
            raise ValueError(
                f"availability must have shape ({self.n_receptors},), got {tuple(availability.shape)}"
            )
        if torch.any((availability < 0) | (availability > 1)):
            raise ValueError("availability must lie in [0, 1]")

        loaded_depot = state.depot + dose
        absorbed = self.p.absorption * loaded_depot
        depot = loaded_depot - absorbed
        concentration = self.p.elimination * state.concentration + absorbed

        drive = (self.p.affinity @ concentration).clamp_min(0)
        drive_power = drive.pow(self.p.hill)
        occupancy = drive_power / (self.p.ec50.pow(self.p.hill) + drive_power)

        sensitivity = (
            state.sensitivity
            + self.p.recovery * (1 - state.sensitivity)
            - self.p.desensitization * occupancy * state.sensitivity
        ).clamp(0, 1)
        adaptation = (
            self.p.homeostasis_decay * state.adaptation
            + float(self.p.homeostasis_gain) * occupancy
        )
        receptor_effect = availability * self.p.emax * (
            sensitivity * occupancy - adaptation
        )
        delta_h = self.p.receptor_basis @ receptor_effect

        return PKPDStep(
            state=PKPDState(concentration, depot, sensitivity, adaptation),
            occupancy=occupancy,
            receptor_effect=receptor_effect,
            delta_h=delta_h,
        )

