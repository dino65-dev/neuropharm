"""CPU simulation of the minimal NeuroPharm PK/PD causal chain.

This validates the artificial pharmacology dynamics only. It is not evidence
of a behavioral effect in a transformer until ``delta_h`` is intervened on a
model and outcomes are measured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pharmacokinetics.state_space import NeuropharmacologyModel, PKPDParameters


def make_single_receptor_model(
    d_model: int,
    half_life: float,
    ec50: float,
    hill: float,
    desensitization: float,
    recovery: float,
    homeostasis_gain: float,
) -> NeuropharmacologyModel:
    direction = torch.zeros(d_model, dtype=torch.float64)
    direction[0] = 1.0
    rho = 2.0 ** (-1.0 / half_life)
    params = PKPDParameters(
        receptor_basis=direction[:, None],
        affinity=torch.ones(1, 1, dtype=torch.float64),
        ec50=torch.tensor([ec50], dtype=torch.float64),
        hill=torch.tensor([hill], dtype=torch.float64),
        emax=torch.ones(1, dtype=torch.float64),
        elimination=torch.tensor([rho], dtype=torch.float64),
        absorption=torch.ones(1, dtype=torch.float64),
        recovery=torch.tensor([recovery], dtype=torch.float64),
        desensitization=torch.tensor([desensitization], dtype=torch.float64),
        homeostasis_decay=0.95,
        homeostasis_gain=homeostasis_gain,
    )
    return NeuropharmacologyModel(params)


def protocol_dose(protocol: str, t: int, dose: float, stop: int) -> float:
    if protocol == "bolus":
        return dose if t == 0 else 0.0
    if protocol == "infusion":
        return dose if t < stop else 0.0
    if protocol == "pulses":
        return dose if t < stop and t % 8 == 0 else 0.0
    raise ValueError(protocol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=["bolus", "infusion", "pulses"], default="pulses")
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--stop", type=int, default=40)
    parser.add_argument("--dose", type=float, default=1.0)
    parser.add_argument("--half-life", type=float, default=12.0)
    parser.add_argument("--ec50", type=float, default=1.0)
    parser.add_argument("--hill", type=float, default=2.0)
    parser.add_argument("--desensitization", type=float, default=0.08)
    parser.add_argument("--recovery", type=float, default=0.02)
    parser.add_argument("--homeostasis-gain", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("artifacts/pkpd/minimal_chain.json"))
    args = parser.parse_args()

    torch.manual_seed(0)
    model = make_single_receptor_model(
        d_model=8,
        half_life=args.half_life,
        ec50=args.ec50,
        hill=args.hill,
        desensitization=args.desensitization,
        recovery=args.recovery,
        homeostasis_gain=args.homeostasis_gain,
    )
    state = model.initial_state()
    rows = []
    for t in range(args.tokens):
        dose = torch.tensor(
            [protocol_dose(args.protocol, t, args.dose, args.stop)],
            dtype=torch.float64,
        )
        step = model.step(state, dose)
        state = step.state
        rows.append({
            "token": t,
            "dose": float(dose[0]),
            "concentration": float(state.concentration[0]),
            "occupancy": float(step.occupancy[0]),
            "sensitivity": float(state.sensitivity[0]),
            "adaptation": float(state.adaptation[0]),
            "receptor_effect": float(step.receptor_effect[0]),
            "delta_h_norm": float(step.delta_h.norm()),
        })

    payload = {
        "status": "dynamics_only_not_transformer_behavioral_evidence",
        "seed": 0,
        "protocol": args.protocol,
        "parameters": vars(args) | {"output": str(args.output)},
        "shapes": {"dose": [1], "receptors": [1], "delta_h": [8]},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

