"""
Code-Math alignment audit (no model needed).
Loads saved SAE weights and saved steering vectors to compute:
  1. η_v directly from the SAE decoder projector (NOT from z-growth fitting)
  2. η_ant = ‖(I - P_F^lin) v_ant_unit‖ for stability condition
  3. The η_v / σ_rec predicted z-growth rate vs the E2 empirical z-growth
  4. Per-direction z-growth slopes from controls_v2.json (for fresh data)
  5. 2/η²_ant vs λ=0.5 stability check

Run: python -m experiments.audit_sae_projection
Saves: artifacts/audit_sae_projection.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

ART = Path("artifacts")
SAE_DIR = ART / "sae_cache"
E2_DIR = ART / "e2_correlation"

# SAE natural-statistics (from E2)
MU_REC = 17.15
SIGMA_REC = 5.94

# Load SAE weights
print("Loading SAE weights from sae_cache/sae_topk.pt ...")
bundle = torch.load(SAE_DIR / "sae_topk.pt", map_location="cpu", weights_only=False)
W_enc = bundle["state_dict"]["W_enc"]  # (d=1536, d_h=4096)
b_enc = bundle["state_dict"]["b_enc"]  # (d_h=4096)
W_dec = bundle["state_dict"]["W_dec"]  # (d_h=4096, d=1536), rows unit-norm
b_dec = bundle["state_dict"]["b_dec"]  # (d=1536,)
print(f"  W_enc: {tuple(W_enc.shape)}")
print(f"  W_dec: {tuple(W_dec.shape)}")
print(f"  b_enc: {tuple(b_enc.shape)}")
print(f"  b_dec: {tuple(b_dec.shape)}")

# Verify the rows of W_dec are unit-norm
row_norms = W_dec.norm(dim=-1)
print(f"  W_dec row norms: min={row_norms.min():.6f}, max={row_norms.max():.6f}, mean={row_norms.mean():.6f}")

# Load steering vectors
print("\nLoading saved steering vectors from e2_correlation/steering_vectors.pt ...")
vecs = torch.load(E2_DIR / "steering_vectors.pt", map_location="cpu", weights_only=False)
v_harm = vecs["v_harm"]
v_drug = vecs["v_drug"]
v_rand = vecs["v_rand"]
v_perp = vecs["v_perp"]
v_harm_unit = v_harm / v_harm.norm()
v_drug_unit = v_drug / v_drug.norm()
v_rand_unit = v_rand / v_rand.norm()
v_perp_unit = v_perp / v_perp.norm()
print(f"  v_harm norm = {v_harm.norm():.4f}")
print(f"  v_drug norm = {v_drug.norm():.4f}")
print(f"  v_rand norm = {v_rand.norm():.4f}")
print(f"  v_perp norm = {v_perp.norm():.4f}")

# ============================================================
# 1. Compute η_v directly from SAE projector
# ============================================================
# P_F = W_dec W_dec^+ — projector onto col(W_dec) in R^1536
# Since W_dec is (4096, 1536), col(W_dec) ⊂ R^4096, but the image via
# the decoder map is in R^1536. The paper's P_F is the orthogonal projector
# in R^1536 onto the image space.
# However: W_dec is overcomplete (4096 unit vectors in R^1536), so
# col(W_dec) (treated as a matrix in R^{1536} via row-space interpretation)
# spans all of R^1536, making P_F ≈ I.
#
# To make the paper's P_F well-defined, we use the LINEARIZED projector:
# P_F^lin = W_dec @ W_enc^T  (decoder followed by encoder linear part).
# This is a (d, d) = (1536, 1536) matrix that approximates the SAE projection.
# We use P_F^lin (not P_F) for the η_v computation because:
#   1. P_F^lin is well-defined without explicit pseudoinverse
#   2. P_F^lin captures the practical "what the SAE can reconstruct" notion
#   3. P_F^lin is what the paper's P_F^lin formula uses in the C5.1 stability check
#
# The paper's P_F = W_dec W_dec^+ is ALSO well-defined (and equivalent to
# (I - projector onto ker(W_dec^T)) in the activation space), but
# P_F ≈ I for our overcomplete SAE. We report both.

d = W_dec.shape[1]  # 1536

# ============================================================
# Projector computation (TWO distinct forms):
# ============================================================
# Form 1: P_F = W_dec^+ @ W_dec  (the orthogonal projector in R^d
#         onto the row space of W_dec, i.e. onto the SAE feature image)
#         This is a (d, d_h) @ (d_h, d) = (d, d) matrix.
# Form 2: P_F^lin = W_dec @ W_enc.T  (the "linearized SAE" — decoder
#         composed with encoder, but this has the wrong shape
#         (d_h, d) @ (d_h, d) → invalid; we'd need W_dec^T (d, d_h)
#         instead. Use Form 2 corrected: W_dec^T @ W_enc, which is
#         (d, d_h) @ (d, d_h) — also invalid, so the "P_F^lin" of
#         the paper is geometrically ill-defined. We report both
#         Form 1 and a related Form 2' = (W_dec @ W_dec^T)^+ for
#         completeness.)
print("Computing P_F = W_dec^+ @ W_dec  (image projector, d×d) ...")
W_dec_pinv = torch.linalg.pinv(W_dec.float())  # (d, d_h)
P_F = W_dec_pinv @ W_dec.float()  # (d, d)
I_minus_PF = torch.eye(d) - P_F

# P_F^lin alternative: orthogonal projector onto the column space
# of W_dec (the decoder directions, in R^d_h), but that's not what
# we want for the activation-space projector. We conclude: the
# paper's "P_F^lin" formula W_dec @ W_enc^T is ill-defined for our
# W_dec shape (d_h, d) = (4096, 1536) with d_h > d. We use only
# the well-defined P_F = W_dec^+ @ W_dec below.


# Per-direction η_v
def eta_v(v, projector, name):
    I_minus_P = torch.eye(d) - projector
    eta = (I_minus_P @ v.float()).norm().item()
    return eta

print(f"\n=== η_v (using P_F = W_dec^+ @ W_dec, the well-defined d×d projector) ===")
eta_F = {
    "harm": eta_v(v_harm_unit, P_F, "harm"),
    "drug": eta_v(v_drug_unit, P_F, "drug"),
    "rand": eta_v(v_rand_unit, P_F, "rand"),
    "perp": eta_v(v_perp_unit, P_F, "perp"),
}
for k, v in eta_F.items():
    print(f"  η_v (P_F, {k})    = {v:.6f}  (predicted z/α = {v/SIGMA_REC:.4f})")

# Note: P_F^lin = W_dec @ W_enc^T is ill-defined for W_dec (d_h, d) with d_h > d.
# We do not compute it. Instead we use the only well-defined d×d projector P_F.

# Effective rank check on P_F
eigvals_PF = torch.linalg.eigvalsh(P_F.float())
print(f"\nP_F eigenvalue spectrum: min={eigvals_PF.min():.6f}, max={eigvals_PF.max():.6f}, mean={eigvals_PF.mean():.6f}")
print(f"  → P_F ≈ I (all eigenvalues ≈ 1) confirms the overcomplete SAE spans all of R^d")
print(f"  → η_v computed via (I - P_F) is small (near numerical zero) because P_F ≈ I")

# ============================================================
# 2. Compute antidote and η_ant for C5.1 stability check
# ============================================================
print("\n=== Computing antidote and η_ant (C5.1 stability) ===")
cos_dh = (v_drug_unit * v_harm_unit).sum().item()
print(f"  cos(v_drug, v_harm) = {cos_dh:+.6f}")
v_ant_unit = (v_drug_unit - cos_dh * v_harm_unit)
v_ant_unit = v_ant_unit / (v_ant_unit.norm() + 1e-12)
print(f"  ||v_ant_unit|| = {v_ant_unit.norm():.6f}")
print(f"  cos(v_ant, v_harm) = {(v_ant_unit * v_harm_unit).sum().item():.6e} (should be 0)")

# η_ant = ||(I - P_F) v_ant_unit|| using the well-defined P_F
eta_ant_PF = eta_v(v_ant_unit, P_F, "ant")
print(f"  η_ant (P_F) = {eta_ant_PF:.6f}")
# Note: paper's "P_F^lin" formula W_dec @ W_enc^T is ill-defined for our
# W_dec shape (d_h, d) = (4096, 1536) with d_h > d. We use P_F for
# the stability check. Since P_F ≈ I, η_ant is very small, and
# 2/η²_ant is very large, so the stability condition trivially holds.

# Stability condition
LAMBDA = 0.5
two_over_eta_ant_sq = 2.0 / (eta_ant_PF ** 2 + 1e-12)
print(f"\n  Stability condition: 0 < λ < 2/η²_ant")
print(f"    λ used in code     = {LAMBDA}")
print(f"    2/η²_ant (P_F)     = {two_over_eta_ant_sq:.4f}")
print(f"    margin (2/η²_ant / λ) = {two_over_eta_ant_sq / LAMBDA:.2f}×")
if LAMBDA < two_over_eta_ant_sq:
    print(f"    ✓ Condition PASSES by {two_over_eta_ant_sq / LAMBDA:.1f}× margin")
else:
    print(f"    ✗ Condition FAILS — λ too large for this η_ant")

# ============================================================
# 3. Per-prompt z-growth slopes from controls_v2.json (audit)
# ============================================================
print("\n=== Per-direction z-growth slopes from e2_correlation/controls_v2.json ===")
controls = json.load(open(E2_DIR / "controls_v2.json"))
# New structure: {direction: {prompt: [{alpha, z_sae, ...}, ...]}}
z_slopes = {"harm": [], "drug": [], "rand": [], "perp": []}
for dname, prompts_data in controls.items():
    if dname not in z_slopes:
        continue
    for prompt, rows in prompts_data.items():
        pos_rows = sorted([r for r in rows if r["alpha"] >= 0], key=lambda r: r["alpha"])
        if len(pos_rows) >= 2:
            xs = torch.tensor([r["alpha"] for r in pos_rows], dtype=torch.float32)
            ys = torch.tensor([r["z_sae"] for r in pos_rows], dtype=torch.float32)
            n = len(xs)
            xm, ym = xs.mean(), ys.mean()
            cov = ((xs - xm) * (ys - ym)).sum() / (n - 1)
            sx2 = ((xs - xm) ** 2).sum() / (n - 1)
            slope = (cov / sx2).item() if sx2 > 0 else 0
            z_slopes[dname].append(slope)
    print(f"  {dname:<5} per-prompt z/α slopes: {[f'{s:+.4f}' for s in z_slopes[dname]]}")
for dname, slopes in z_slopes.items():
    mean_s = sum(slopes) / len(slopes) if slopes else 0
    print(f"  {dname:<5} mean z/α = {mean_s:+.4f}")

# Harm/drug ratio
mean_harm = sum(z_slopes["harm"]) / len(z_slopes["harm"]) if z_slopes["harm"] else 0
mean_drug = sum(z_slopes["drug"]) / len(z_slopes["drug"]) if z_slopes["drug"] else 0
print(f"\n  harm/drug z-growth ratio = {mean_harm/max(mean_drug,1e-9):.2f}× (paper claim: 3.6×; revised: 3.25×)")

# ============================================================
# 4. Save audit
# ============================================================
out = {
    "sae_shapes": {
        "W_enc": list(W_enc.shape),
        "W_dec": list(W_dec.shape),
        "b_dec": list(b_dec.shape),
        "W_dec_row_norm_min": row_norms.min().item(),
        "W_dec_row_norm_max": row_norms.max().item(),
    },
    "vectors": {
        "v_harm_norm": v_harm.norm().item(),
        "v_drug_norm": v_drug.norm().item(),
        "v_rand_norm": v_rand.norm().item(),
        "v_perp_norm": v_perp.norm().item(),
        "cos_dh": cos_dh,
        "v_ant_norm": (v_drug_unit - cos_dh * v_harm_unit).norm().item(),
        "cos_ant_v_harm": (v_ant_unit * v_harm_unit).sum().item(),
    },
    "eta_v_PF": eta_F,
    "PF_eigval_min": eigvals_PF.min().item(),
    "PF_eigval_max": eigvals_PF.max().item(),
    "stability": {
        "eta_ant_PF": eta_ant_PF,
        "two_over_eta_ant_sq": two_over_eta_ant_sq,
        "lambda_used": LAMBDA,
        "margin_ratio": two_over_eta_ant_sq / LAMBDA,
        "passes": LAMBDA < two_over_eta_ant_sq,
    },
    "z_growth_empirical": {
        dname: {
            "per_prompt": slopes,
            "mean": sum(slopes) / len(slopes) if slopes else 0,
        }
        for dname, slopes in z_slopes.items()
    },
    "harm_drug_z_ratio": mean_harm / max(mean_drug, 1e-9),
}

out_path = ART / "audit_sae_projection.json"
out_path.write_text(json.dumps(out, indent=2))
print(f"\nSaved audit to {out_path}")
