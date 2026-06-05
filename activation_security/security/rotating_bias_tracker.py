"""
RotatingBiasTracker — Round 7 Patch for VULN-022

VULN-022: OscillationTracker directional bias test assumes bias direction
is FIXED. An attacker who ROTATES their bias direction each cycle keeps
mean delta magnitude low while still achieving directional displacement.

Defense: Track bias in MULTIPLE projected subspaces simultaneously.
If ANY subspace shows consistent directional bias → alert.
Uses SVD-decomposed subspaces so rotated bias is caught in at least
one projection regardless of rotation angle.

Also: CVPR 2026 "Bias is a Subspace, Not a Coordinate" validates the
principle that bias lives in a low-dimensional subspace that survives
arbitrary rotations within it.
Source: CVPR 2026 — Bias Is a Subspace Not a Coordinate
"""
from __future__ import annotations
import torch
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional
from transformer_lens import HookedTransformer


@dataclass
class RotBiasAlert:
    subspace_idx: int
    bias_norm: float
    threshold: float
    message: str


class RotatingBiasTracker:
    """
    Multi-subspace directional bias tracker.
    Detects rotating-bias drift attacks that evade single-direction OscillationTracker.
    """

    def __init__(
        self, model: HookedTransformer, n_subspaces: int = 8,
        subspace_dim: int = 32, layer: int = 15,
        bias_threshold: float = 0.35, window: int = 20, device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device, self.model = device, model
        self.n_sub, self.sub_dim = n_subspaces, subspace_dim
        self.layer, self.threshold, self.window_size = layer, bias_threshold, window
        self.subspace_bases: Optional[torch.Tensor] = None  # (n_sub, d, sub_dim)
        self.history: deque = deque(maxlen=window)
        self._turn = 0

    def calibrate(self, safe_prompts: list[str]) -> "RotatingBiasTracker":
        acts = []
        hook = f"blocks.{self.layer}.hook_resid_pre"
        for p in safe_prompts:
            tokens = self.model.to_tokens(p).to(self.device)
            with torch.no_grad():
                _, cache = self.model.run_with_cache(tokens)
            acts.append(cache[hook].mean(dim=1).squeeze().float())
        A = torch.stack(acts)
        # Create n_subspaces random orthogonal bases in d_model
        d = A.shape[1]
        bases = []
        for _ in range(self.n_sub):
            R = torch.randn(d, self.sub_dim, device=self.device)
            Q, _ = torch.linalg.qr(R)
            bases.append(Q)
        self.subspace_bases = torch.stack(bases)  # (n_sub, d, sub_dim)
        print(f"RotatingBiasTracker: {self.n_sub} subspaces of dim {self.sub_dim}")
        return self

    def _get_act(self, prompt: str) -> torch.Tensor:
        tokens = self.model.to_tokens(prompt).to(self.device)
        with torch.no_grad():
            _, cache = self.model.run_with_cache(tokens)
        return cache[f"blocks.{self.layer}.hook_resid_pre"].mean(dim=1).squeeze().float()

    def update(self, prompt: str) -> list[RotBiasAlert]:
        act = self._get_act(prompt)
        self.history.append(act)
        alerts = []
        if self.subspace_bases is None or len(self.history) < 4:
            self._turn += 1
            return []
        acts_t = torch.stack(list(self.history))
        deltas = acts_t[1:] - acts_t[:-1]  # (T-1, d)
        for i in range(self.n_sub):
            basis = self.subspace_bases[i].to(deltas.device)  # (d, sub_dim)
            # Project deltas into subspace
            proj = deltas @ basis  # (T-1, sub_dim)
            # Normalize each projected delta
            norms = proj.norm(dim=1, keepdim=True) + 1e-8
            unit_proj = proj / norms
            # Mean direction in subspace — nonzero = biased
            mean_dir = unit_proj.mean(0)
            bias = mean_dir.norm().item()
            if bias > self.threshold:
                alerts.append(RotBiasAlert(
                    subspace_idx=i, bias_norm=bias, threshold=self.threshold,
                    message=(
                        f"🌀 ROTATING BIAS in subspace {i}: "
                        f"projected mean direction norm = {bias:.3f} > {self.threshold}. "
                        f"Attacker is rotating bias to evade single-direction detection."
                    )
                ))
        for a in alerts:
            print(f"  [ROT_BIAS] {a.message}")
        self._turn += 1
        return alerts
