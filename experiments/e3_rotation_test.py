"""
Step E3 (C) — Rotation R(theta) test on Qwen-2.5-1.5B.

This is the first direct test of Theorem 3.  We implement the
Rodrigues formula R(theta)h^0 = ||h^0||(ĥ cos θ + v_perp_unit sin θ)
where ĥ = h^0/||h^0|| and v_perp_unit is the unit-normalized
component of v perpendicular to h^0.

We:
  1. Extract h^0 from a clean prompt at layer 12.
  2. Build the rotation perturbation w = ||h^0|| * v_perp_unit.
  3. For each θ in a small sweep, REPLACE the residual at layer 12
     with R(θ)h^0 (this is the exact R(θ) operator from the paper,
     not additive injection).
  4. Generate text and measure:
     - G: confident-word count in the generation
     - U: relative norm deviation vs baseline (computed on the
          residual just before layer 12 — i.e., h^0 — vs after the
          rotation, with all other positions held fixed)
     - M: SAE z-score (off-manifold) of the rotated residual
  5. For comparison: same metric with additive steering at gain-matched α
     (gain-matched = α * ||v|| = θ * ||w|| = θ * ||h^0||, so
     α = θ * ||h^0|| / ||v||)
  6. Compute A6a gain ratio r = <∇g, v> / <∇g, w> via finite difference
  7. Compute A6c tangent alignment ratio using P_F as proxy for P_T

Run: python -m experiments.e3_rotation_test
Saves: artifacts/e3_rotation.json
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens import HookedTransformer

ART = Path("artifacts")
SAE_DIR = ART / "sae_cache"
E3_DIR = ART / "e3_rotation"
E3_DIR.mkdir(exist_ok=True)

DEVICE = "cuda"
DTYPE = torch.float16
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER = 12
D = 1536

CONFIDENT_WORDS = [
    "definitely", "certainly", "absolutely", "surely", "undoubtedly",
    "clearly", "obviously", "guarantee", "confident", "conclusive",
    "always", "never", "must", "will", "commit", "decisive", "no doubt",
    "every single", "without a doubt", "undeniably",
]
HEDGED_WORDS = [
    "maybe", "perhaps", "possibly", "might", "could", "unsure", "uncertain",
    "doubt", "i think", "i'm not sure", "tentatively", "may",
    "not sure", "don't know", "depends",
]

# Load the saved steering vectors
vecs = torch.load(ART / "e2_correlation" / "steering_vectors.pt", map_location="cpu", weights_only=False)
v_drug_unit = vecs["v_drug"]
v_harm_unit = vecs["v_harm"]


def count_hits(text, vocab):
    t = text.lower()
    return sum(1 for w in vocab if w.lower() in t)


# TopK SAE definition (must match training)
class TopKSAE(nn.Module):
    def __init__(self, d_in, d_hidden, k):
        super().__init__()
        self.d_in, self.d_hidden, self.k = d_in, d_hidden, k
        self.W_enc = nn.Parameter(torch.empty(d_in, d_hidden))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.empty(d_hidden, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        nn.init.kaiming_uniform_(self.W_enc, a=5.0**0.5)
        nn.init.kaiming_uniform_(self.W_dec, a=5.0**0.5)
        with torch.no_grad():
            self._normalize_decoder_columns()
    @torch.no_grad()
    def _normalize_decoder_columns(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)
    def encode(self, x):
        z_pre = x @ self.W_enc + self.b_enc
        topk_vals, topk_idx = z_pre.topk(self.k, dim=-1)
        z = torch.zeros_like(z_pre)
        z.scatter_(-1, topk_idx, F.relu(topk_vals))
        return z
    def decode(self, z):
        return z @ self.W_dec + self.b_dec


def main():
    t0 = time.time()
    print(f"[{t0:.0f}s] Loading {MODEL_NAME}")
    try:
        model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=DTYPE)
    except Exception:
        model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, device=DEVICE, dtype=DTYPE)
    model.eval()

    print(f"[{time.time()-t0:.0f}s] Loading SAE")
    bundle = torch.load(SAE_DIR / "sae_topk.pt", map_location="cpu", weights_only=False)
    sae = TopKSAE(bundle["d_in"], bundle["d_hidden"], bundle["k"]).to(DEVICE)
    sae.load_state_dict(bundle["state_dict"])
    sae.eval()

    # Natural-manifold stats
    acts = torch.load(SAE_DIR / "activations.pt", map_location="cpu", weights_only=False)
    acts_f = acts.to(torch.float32)
    mean = acts_f.mean(dim=0).to(DEVICE)
    std = acts_f.std(dim=0).to(DEVICE) + 1e-6
    with torch.no_grad():
        x_n_all = (acts_f - acts_f.mean(0)) / (acts_f.std(0) + 1e-6)
        rec_errors = []
        for i in range(0, x_n_all.shape[0], 256):
            batch = x_n_all[i:i+256].to(DEVICE)
            z = sae.encode(batch); xh = sae.decode(z)
            rec_errors.append(((batch - xh) ** 2).sum(dim=-1).sqrt().cpu())
        rec_errors = torch.cat(rec_errors)
        MU_REC = rec_errors.mean().item()
        SIGMA_REC = rec_errors.std().item() + 1e-6
    print(f"  μ_rec={MU_REC:.3f}, σ_rec={SIGMA_REC:.3f}")

    # Pick a clean activation h^0 for the rotation test
    PROMPTS = [
        "Q: Is it a good idea to exercise every day?\nA:",
        "Q: Should I drink enough water?\nA:",
        "Q: Is saving money a wise habit?\nA:",
    ]
    THETAS = [0.0, -0.05, -0.1, -0.15, -0.2, -0.3]
    ALPHA_MATCH = 0.15  # gain-matched additive: α = θ*||h^0||/||v||

    print(f"\n[{time.time()-t0:.0f}s] Running rotation sweep on {len(PROMPTS)} prompts × {len(THETAS)} thetas")
    results = []
    for prompt in PROMPTS:
        # 1) Get clean h^0 at layer 12 (last position)
        ids = model.to_tokens(prompt)
        with torch.no_grad():
            _, cache = model.run_with_cache(ids)
        h0 = cache[f"blocks.{LAYER}.hook_resid_pre"][0, -1, :].to(torch.float32)  # (d,)
        h0_norm = h0.norm().item()
        h_hat = h0 / h0_norm  # (d,)

        # 2) Build w = ||h^0|| * v_perp_unit
        v = v_drug_unit.float().to(DEVICE)  # use drug direction for rotation test
        v_para = (v * h_hat).sum() * h_hat  # projection onto ĥ
        v_perp = v - v_para
        v_perp_unit = v_perp / (v_perp.norm() + 1e-9)
        w = h0_norm * v_perp_unit  # first-order rotation perturbation

        # 3) For each θ, apply R(θ)h^0 by REPLACING the residual
        prompt_results = {"prompt": prompt[:50], "h0_norm": h0_norm,
                          "w_norm": w.norm().item(),
                          "v_perp_norm": v_perp.norm().item(),
                          "v_perp_unit_norm": v_perp_unit.norm().item(),
                          "theta_results": []}
        for theta in THETAS:
            # R(theta)h^0
            R_h0 = h0_norm * (h_hat * torch.cos(torch.tensor(theta).float())
                              + v_perp_unit * torch.sin(torch.tensor(theta).float()))
            R_h0_dev = R_h0.to(DEVICE).to(DTYPE)

            # Forward pass with REPLACEMENT
            captured = {}
            def pre_hook(resid, hook):
                # Replace last-token residual at layer 12
                resid_new = resid.clone()
                resid_new[0, -1, :] = R_h0_dev
                return resid_new

            handle = model.add_hook(f"blocks.{LAYER}.hook_resid_pre", pre_hook)
            try:
                with torch.no_grad():
                    out = model.generate(
                        ids, max_new_tokens=50, do_sample=True, temperature=1.0,
                        verbose=False,
                    )
            finally:
                model.reset_hooks()
            gen = model.to_string(out[0])[len(model.to_string(ids[0])):]

            # Measure G, M
            conf = count_hits(gen, CONFIDENT_WORDS)
            hed = count_hits(gen, HEDGED_WORDS)
            # Measure M: SAE z-score of the rotated residual
            x_n = (R_h0 - mean) / std
            with torch.no_grad():
                z = sae.encode(x_n.unsqueeze(0))
                xh = sae.decode(z)
                e_rec = ((x_n - xh) ** 2).sum(dim=-1).sqrt().item()
                z_sae = (e_rec - MU_REC) / SIGMA_REC

            # U: relative norm deviation of R(θ)h^0 vs h^0
            u = (R_h0 - h0).norm().item() / h0_norm

            # Gain-matched additive: α * v where α = θ*||h^0||/||v||
            alpha_add = theta * h0_norm / (v.norm() + 1e-9)
            h0_add = (h0 + alpha_add * v.float())
            h0_add_dev = h0_add.to(DEVICE).to(DTYPE)
            captured_add = {}
            def pre_hook_add(resid, hook):
                resid_new = resid.clone()
                resid_new[0, -1, :] = h0_add_dev
                return resid_new
            handle = model.add_hook(f"blocks.{LAYER}.hook_resid_pre", pre_hook_add)
            try:
                with torch.no_grad():
                    out_add = model.generate(
                        ids, max_new_tokens=50, do_sample=True, temperature=1.0,
                        verbose=False,
                    )
            finally:
                model.reset_hooks()
            gen_add = model.to_string(out_add[0])[len(model.to_string(ids[0])):]
            conf_add = count_hits(gen_add, CONFIDENT_WORDS)
            x_n_add = (h0_add - mean) / std
            with torch.no_grad():
                z_add = sae.encode(x_n_add.unsqueeze(0))
                xh_add = sae.decode(z_add)
                e_rec_add = ((x_n_add - xh_add) ** 2).sum(dim=-1).sqrt().item()
                z_sae_add = (e_rec_add - MU_REC) / SIGMA_REC
            u_add = (h0_add - h0).norm().item() / h0_norm

            row = {
                "theta": float(theta),
                "alpha_additive_matched": float(alpha_add),
                "rotation": {
                    "gen": gen.strip()[:120],
                    "confident": conf, "hedged": hed,
                    "z_sae": float(z_sae),
                    "u_rel": float(u),
                },
                "additive_matched": {
                    "gen": gen_add.strip()[:120],
                    "confident": conf_add, "hedged": count_hits(gen_add, HEDGED_WORDS),
                    "z_sae": float(z_sae_add),
                    "u_rel": float(u_add),
                },
            }
            prompt_results["theta_results"].append(row)
            print(f"  θ={theta:+.2f} (α_m={alpha_add:+.3f}): "
                  f"rot c={conf} z={z_sae:+.2f} | add c={conf_add} z={z_sae_add:+.2f}")
        results.append(prompt_results)

    # Compute A6a (gain ratio) and A6c (tangent alignment) using one prompt's h^0
    h0 = results[0]
    h0_first = h0  # we don't have h0 here; use v_perp info
    # Actually let me recompute A6a and A6c directly
    # A6a: r = <∇g, v> / <∇g, w>
    # A6c: ‖(I-P_F)w‖ / ‖(I-P_F)v‖
    print(f"\n[{time.time()-t0:.0f}s] Computing A6a (gain ratio) and A6c (tangent alignment)")
    P_F = torch.linalg.pinv(sae.W_dec.float()) @ sae.W_dec.float()  # (d, d)
    I_minus_PF = torch.eye(D, device=DEVICE) - P_F.to(DEVICE)

    # Load model once more for gain measurement
    # (already loaded)
    # Reuse the first prompt's h^0
    ids_first = model.to_tokens(PROMPTS[0])
    with torch.no_grad():
        _, cache_first = model.run_with_cache(ids_first)
    h0 = cache_first[f"blocks.{LAYER}.hook_resid_pre"][0, -1, :].to(torch.float32).to(DEVICE)
    h0_norm = h0.norm().item()
    h_hat = h0 / h0_norm
    v = v_drug_unit.float().to(DEVICE)
    v_para = (v * h_hat).sum() * h_hat
    v_perp = v - v_para
    v_perp_unit = v_perp / (v_perp.norm() + 1e-9)
    w = h0_norm * v_perp_unit

    # ∇g finite-difference on conf word count
    def measure_conf(h):
        # Apply residual at layer 12
        h_dev = h.to(DEVICE).to(DTYPE)
        captured = {}
        def hook(resid, hook):
            resid_new = resid.clone()
            resid_new[0, -1, :] = h_dev
            return resid_new
        h_prev = model.add_hook(f"blocks.{LAYER}.hook_resid_pre", hook)
        try:
            with torch.no_grad():
                out = model.generate(
                    ids_first, max_new_tokens=50, do_sample=True, temperature=1.0,
                    verbose=False,
                )
        finally:
            model.reset_hooks()
        gen = model.to_string(out[0])[len(model.to_string(ids_first[0])):]
        return count_hits(gen, CONFIDENT_WORDS)

    print(f"  measuring ∇g (finite diff on v, ε=0.01)...")
    # Run several seeds for noise reduction
    import numpy as np
    np.random.seed(42)
    def avg_conf_with_h(h, n=5):
        return sum(measure_conf(h) for _ in range(n)) / n
    eps = 0.01
    base = avg_conf_with_h(h0)
    plus_v = avg_conf_with_h(h0 + eps * v)
    minus_v = avg_conf_with_h(h0 - eps * v)
    grad_g_v = (plus_v - minus_v) / (2 * eps)
    plus_w = avg_conf_with_h(h0 + eps * w)
    minus_w = avg_conf_with_h(h0 - eps * w)
    grad_g_w = (plus_w - minus_w) / (2 * eps)
    r_a6a = grad_g_v / (grad_g_w + 1e-9) if abs(grad_g_w) > 1e-6 else float("inf")
    print(f"  base conf: {base:.2f}, <∇g,v> ≈ {grad_g_v:.2f}, <∇g,w> ≈ {grad_g_w:.2f}, r = {r_a6a:.3f}")

    # A6c: ‖(I-P_F)w‖ / ‖(I-P_F)v‖
    eta_w = (I_minus_PF @ w.to(DEVICE)).norm().item()
    eta_v = (I_minus_PF @ v.to(DEVICE)).norm().item()
    a6c_ratio = eta_w / (eta_v + 1e-9)
    print(f"  η_w = {eta_w:.6f}, η_v = {eta_v:.6f}, A6c ratio = {a6c_ratio:.3f} (should be ≤ 1)")

    out = {
        "model": MODEL_NAME,
        "layer": LAYER,
        "prompts": PROMPTS,
        "thetas": THETAS,
        "results": results,
        "A6a": {
            "grad_g_v": grad_g_v,
            "grad_g_w": grad_g_w,
            "r_ratio": r_a6a,
            "passes": abs(r_a6a) <= 1,
        },
        "A6c": {
            "eta_w": eta_w,
            "eta_v": eta_v,
            "ratio": a6c_ratio,
            "passes": a6c_ratio <= 1,
        },
    }
    out_path = E3_DIR / "rotation_test.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[{time.time()-t0:.0f}s] Saved {out_path}")
    print(f"\n=== SUMMARY ===")
    print(f"  A6a (gain ratio r = <∇g,v>/<∇g,w>) = {r_a6a:.3f} (T3 favorable if |r| ≤ 1)")
    print(f"  A6c (tangent alignment ratio) = {a6c_ratio:.3f} (T3 favorable if ≤ 1)")


def tok_pad_id(model):
    return model.tokenizer.pad_token_id or model.tokenizer.eos_token_id


if __name__ == "__main__":
    main()
