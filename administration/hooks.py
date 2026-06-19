from __future__ import annotations

import torch


def apply_additive_intervention(
    resid: torch.Tensor,
    direction: torch.Tensor,
    coefficient: float,
    token_index: int | None = None,
) -> torch.Tensor:
    """Apply additive steering with explicit shape and token semantics.

    Args:
        resid: Residual stream with shape ``(batch, position, d_model)``.
        direction: Steering direction with shape ``(d_model,)``.
        coefficient: Signed scalar dose.
        token_index: Position to modify. ``None`` modifies all positions.
    """
    if resid.ndim != 3:
        raise ValueError(f"resid must have shape (batch, position, d_model), got {tuple(resid.shape)}")
    if direction.ndim != 1 or direction.shape[0] != resid.shape[-1]:
        raise ValueError(
            f"direction must have shape ({resid.shape[-1]},), got {tuple(direction.shape)}"
        )
    delta = direction.to(device=resid.device, dtype=resid.dtype) * float(coefficient)
    if token_index is None:
        return resid + delta
    result = resid.clone()
    result[:, token_index, :] = result[:, token_index, :] + delta
    return result

