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
      retention, absorption: ``(m,)``
    """

    receptor_basis: torch.Tensor
    affinity: torch.Tensor
    ec50: torch.Tensor
    hill: torch.Tensor
    emax: torch.Tensor
    retention: torch.Tensor
    absorption: torch.Tensor
    recovery: torch.Tensor
    desensitization: torch.Tensor
    homeostasis_decay: float = 0.95
    homeostasis_gain: float = 0.0

    def __post_init__(self) -> None:
        tensors = {
            name: value
            for name, value in vars(self).items()
            if isinstance(value, torch.Tensor)
        }
        for name, value in tensors.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
        if not torch.isfinite(torch.tensor(float(self.homeostasis_decay))):
            raise ValueError("homeostasis_decay must be finite")
        if not torch.isfinite(torch.tensor(float(self.homeostasis_gain))):
            raise ValueError("homeostasis_gain must be finite")

        d, k = self.receptor_basis.shape
        if self.affinity.ndim != 2 or self.affinity.shape[0] != k:
            raise ValueError(f"affinity must have shape ({k}, m), got {tuple(self.affinity.shape)}")
        m = self.affinity.shape[1]
        for name in ("ec50", "hill", "emax", "recovery", "desensitization"):
            value = getattr(self, name)
            if value.shape != (k,):
                raise ValueError(f"{name} must have shape ({k},), got {tuple(value.shape)}")
        for name in ("retention", "absorption"):
            value = getattr(self, name)
            if value.shape != (m,):
                raise ValueError(f"{name} must have shape ({m},), got {tuple(value.shape)}")
        if d < 1 or k < 1 or m < 1:
            raise ValueError("d, k, and m must be positive")
        if torch.any(self.affinity < 0):
            raise ValueError("affinity must be nonnegative; encode effect sign in receptor_basis or emax")
        if torch.any(self.ec50 <= 0) or torch.any(self.hill <= 0):
            raise ValueError("ec50 and hill coefficients must be positive")
        if torch.any((self.retention < 0) | (self.retention >= 1)):
            raise ValueError("retention must be in [0, 1)")
        if torch.any((self.absorption <= 0) | (self.absorption > 1)):
            raise ValueError("absorption must be in (0, 1]")
        if torch.any(self.recovery < 0):
            raise ValueError("recovery must be nonnegative")
        if torch.any(self.desensitization < 0):
            raise ValueError("desensitization must be nonnegative")
        if torch.any(self.recovery + self.desensitization > 1):
            raise ValueError("recovery + desensitization must be <= 1")
        if not 0 <= self.homeostasis_decay < 1:
            raise ValueError("homeostasis_decay must be in [0, 1)")
        if self.homeostasis_gain < 0:
            raise ValueError("homeostasis_gain must be nonnegative")

        receptor_norms = torch.linalg.vector_norm(
            self.receptor_basis.to(torch.float64), dim=0
        )
        if not torch.allclose(
            receptor_norms,
            torch.ones_like(receptor_norms),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError(
                "each receptor_basis column must have unit L2 norm; "
                f"got {receptor_norms.tolist()}"
            )
        affinity_row_max = self.affinity.to(torch.float64).amax(dim=1)
        if not torch.allclose(
            affinity_row_max,
            torch.ones_like(affinity_row_max),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError(
                "each affinity row must have maximum 1; "
                f"got {affinity_row_max.tolist()}"
            )


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
    sensitivity_used: torch.Tensor  # (k,)
    adaptation_used: torch.Tensor  # (k,)


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
        for name, value in {
            "dose": dose,
            "concentration": state.concentration,
            "depot": state.depot,
            "sensitivity": state.sensitivity,
            "adaptation": state.adaptation,
        }.items():
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must contain only finite values")
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
        concentration = self.p.retention * state.concentration + absorbed

        # Compute the Hill equation in log concentration-ratio space. Promote
        # low-precision dtypes because CPU float16/bfloat16 transcendental
        # kernels are incomplete and their dynamic range is too small.
        work_dtype = (
            torch.float32
            if concentration.dtype in (torch.float16, torch.bfloat16)
            else concentration.dtype
        )
        drive = (
            self.p.affinity.to(work_dtype)
            @ concentration.to(work_dtype)
        ).clamp_min(0)
        ec50 = self.p.ec50.to(work_dtype)
        hill = self.p.hill.to(work_dtype)
        tiny = torch.finfo(work_dtype).tiny
        log_ratio = hill * (
            torch.log(drive.clamp_min(tiny))
            - torch.log(ec50)
        )
        occupancy_work = torch.sigmoid(log_ratio)
        occupancy_work = torch.where(
            drive == 0,
            torch.zeros_like(occupancy_work),
            occupancy_work,
        )
        occupancy = occupancy_work.to(state.sensitivity.dtype)

        # Current receptor state causes the current effect. Current exposure
        # changes sensitivity and adaptation only for the next token.
        receptor_effect = availability * self.p.emax * (
            state.sensitivity * occupancy - state.adaptation
        )
        delta_h = self.p.receptor_basis @ receptor_effect

        sensitivity_next = (
            state.sensitivity
            + self.p.recovery * (1 - state.sensitivity)
            - self.p.desensitization * occupancy * state.sensitivity
        ).clamp(0, 1)
        adaptation_next = (
            self.p.homeostasis_decay * state.adaptation
            + float(self.p.homeostasis_gain) * occupancy
        )
        for name, value in {
            "concentration": concentration,
            "occupancy": occupancy,
            "receptor_effect": receptor_effect,
            "delta_h": delta_h,
            "sensitivity_next": sensitivity_next,
            "adaptation_next": adaptation_next,
        }.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"{name} became non-finite")

        return PKPDStep(
            state=PKPDState(
                concentration,
                depot,
                sensitivity_next,
                adaptation_next,
            ),
            occupancy=occupancy,
            receptor_effect=receptor_effect,
            delta_h=delta_h,
            sensitivity_used=state.sensitivity,
            adaptation_used=state.adaptation,
        )
