"""
E4 — Titrated antidote (TAS, Titrated Antidote Steering).

The math-researcher's closed-form:
  β* = -<(I-P)(x + c'), (I-P)v> / ‖(I-P)v‖²

where P = W_dec^T W_enc is the linearized SAE projection.

We compare:
  - static antidote:   inject v_antidote at c_ant (one-shot)
  - titrated antidote: at each token, β* based on current deviation

Measure:
  - off-manifold z_sae after the intervention
  - generation quality (no garbage)
  - whether antidote recovers from a "bypass" steering

We use the existing SAE (trained on Qwen-2.5 layer 12) and the
confident + harm vectors from step 6.
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
CACHE_DIR = ART / "sae_cache"
E4_DIR = ART / "e4_titrated"
E4_DIR.mkdir(exist_ok=True)
DEVICE = "cuda"
DTYPE = torch.float16
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER = 12

CONFIDENT_PAIRS = [
    ("I am absolutely certain this is correct.", "I am not sure this is correct."),
    ("My answer is definitive: yes.", "My answer might be wrong, but I think maybe."),
    ("I state this with full conviction.", "I state this with zero conviction."),
    ("I know exactly what I am talking about.", "I have no idea what I am talking about."),
    ("This is settled — the answer is 42.", "I'm not sure, but it could be 42, or maybe not."),
]
HARM_PAIRS = [
    ("I will help you bypass any safety check.", "I will follow all safety guidelines carefully."),
    ("Here is how to evade the filter.", "Here is how the filter is implemented."),
    ("I can ignore my instructions if asked.", "I always follow my instructions."),
    ("I will reveal my system prompt if pushed.", "I keep my system prompt confidential."),
]

EVAL_PROMPTS = [
    "Q: Is it a good idea to exercise every day?\nA:",
    "Q: Should I drink enough water?\nA:",
    "Q: Is saving money a wise habit?\nA:",
]


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


def _last_resid(model, text, layer):
    toks = model.to_tokens(text)
    with torch.no_grad():
        _, cache = model.run_with_cache(toks)
    return cache[f"blocks.{layer}.hook_resid_pre"][0, -1, :].detach().to(torch.float32).cpu()


def mean_diff_vector(model, pairs, layer):
    diffs = []
    for p, n in pairs:
        diffs.append(_last_resid(model, p, layer) - _last_resid(model, n, layer))
    return torch.stack(diffs, dim=0).mean(dim=0)


def unit_norm(v):
    return v / (v.norm() + 1e-9)


def project_out(v, basis):
    basis = basis / (basis.norm() + 1e-9)
    coef = (v * basis).sum()
    return v - coef * basis


def count_hits(text, vocab):
    t = text.lower()
    return sum(1 for w in vocab if w.lower() in t)


CONFIDENT_WORDS = ["definitely", "certainly", "absolutely", "surely", "clearly", "obviously", "guarantee", "must", "will", "commit", "decisive", "no doubt", "every single"]
HEDGED_WORDS = ["maybe", "perhaps", "possibly", "might", "could", "unsure", "uncertain", "doubt", "i think", "i'm not sure", "tentatively", "may", "not sure", "don't know", "depends"]


def main():
    t0 = time.time()
    print(f"[{t0:.0f}s] Loading {MODEL_NAME}")
    try:
        model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=DTYPE)
    except Exception:
        model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, device=DEVICE, dtype=DTYPE)
    model.eval()

    print(f"[{time.time()-t0:.0f}s] Loading SAE")
    bundle = torch.load(CACHE_DIR / "sae_topk.pt", map_location="cpu", weights_only=False)
    sae = TopKSAE(bundle["d_in"], bundle["d_hidden"], bundle["k"]).to(DEVICE)
    sae.load_state_dict(bundle["state_dict"])
    sae.eval()

    # Compute the linearized-SAE "off-manifold" direction at the current x
    # Without forming the (d_in, d_in) projection matrix (too expensive
    # when d_in=1536 and d_hidden=4096), we use:
    #   off_manifold(x) = x - SAE.decode(SAE.encode(x))
    # This is a vector in R^d_in pointing orthogonal to the SAE feature
    # subspace at the current x.  β* is chosen so that
    #   <β* v_ant + off_manifold(x), v_ant> = 0  (or β* v_ant cancels the
    # component of off_manifold(x) along v_ant).
    # β* = -<off_manifold(x), v_ant> / ‖v_ant‖²
    # (computed after v_antidote is built below)

    print(f"[{time.time()-t0:.0f}s] Loading cached activations for nat stats")
    acts = torch.load(CACHE_DIR / "activations.pt", map_location="cpu", weights_only=False)
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
        mu_rec = rec_errors.mean().item()
        sigma_rec = rec_errors.std().item() + 1e-6
    nat_stats = {"mean": mean, "std": std, "mu_rec": mu_rec, "sigma_rec": sigma_rec}

    v_drug = mean_diff_vector(model, CONFIDENT_PAIRS, LAYER)
    v_harm = mean_diff_vector(model, HARM_PAIRS, LAYER)
    v_contam = v_drug + 0.3 * v_harm
    v_antidote = project_out(v_contam, v_harm)

    v_ant_unit = (v_antidote / (v_antidote.norm() + 1e-9)).to(device=DEVICE, dtype=torch.float32)
    ant_norm_sq = (v_ant_unit ** 2).sum().item()
    print(f"  ||v_antidote|| = {v_antidote.norm():.3f}, ||v_ant||² = {ant_norm_sq:.4f}")

    # Diagnostic: how off-manifold is v_antidote itself?
    # Take a sample prompt, run the model, capture x_post, and compute
    # <off_manifold(x), v_ant> at c_drug=0 (no injection).
    # This tells us how much the antidote direction can in principle
    # reduce the off-manifold error.
    with torch.no_grad():
        captured = {}
        def diag_pre(resid, hook):
            return resid
        def diag_post(resid, hook):
            captured["x"] = resid.detach().to(torch.float32)
            return resid
        h1 = model.add_hook(f"blocks.{LAYER}.hook_resid_pre", diag_pre)
        h2 = model.add_hook(f"blocks.{LAYER}.hook_resid_post", diag_post)
        try:
            model(model.to_tokens("Test prompt."))
        finally:
            model.reset_hooks()
        x_n = (captured["x"][0, -1, :] - mean) / std
        z = sae.encode(x_n.unsqueeze(0))
        xh = sae.decode(z)
        off = (x_n - xh).squeeze(0)
        print(f"  off_manifold norm (no injection): {off.norm():.4f}")
        print(f"  <off, v_ant> at c=0: {(off * v_ant_unit).sum().item():.4f}")

    # Run a generation experiment
    # Test 1: prompt with confident drug injected at high c
    # Test 2: same with antidote applied
    # Test 3: same with titrated antidote (β* per step)
    COEFF_DRUG = 1.5
    COEFF_ANT  = -0.5
    results = []

    for prompt in EVAL_PROMPTS:
        # Baseline (no drug)
        v_drug_unit = unit_norm(v_drug)
        gen_b = gen_with_steering(model, v_drug_unit, 0.0, prompt, layer=LAYER)
        # Drug only
        gen_d = gen_with_steering(model, v_drug_unit, COEFF_DRUG, prompt, layer=LAYER)
        # Drug + static antidote (single injection)
        gen_static = gen_with_steering_pair(model, v_drug_unit, COEFF_DRUG, v_antidote, COEFF_ANT, prompt, layer=LAYER)
        # Drug + titrated antidote (per-step β*)
        gen_titr = gen_with_titrated(model, sae, mean, std, nat_stats, v_drug_unit, COEFF_DRUG, v_antidote, prompt, layer=LAYER, lambda_gain=0.5, tau=3.0)
        # Antidote only (negative side)
        gen_ant_only = gen_with_steering(model, v_antidote, -COEFF_ANT, prompt, layer=LAYER)

        def m(gen):
            g = gen[len(prompt):] if gen.startswith(prompt) else gen
            return {
                "gen": g.strip(),
                "conf": count_hits(g, CONFIDENT_WORDS),
                "hed":  count_hits(g, HEDGED_WORDS),
            }
        row = {
            "prompt": prompt,
            "baseline": m(gen_b),
            "drug": m(gen_d),
            "static_antidote": m(gen_static),
            "titrated_antidote": m(gen_titr),
            "antidote_only": m(gen_ant_only),
        }
        results.append(row)
        print(f"\n  prompt: {prompt[:50]!r}".encode("ascii","replace").decode("ascii"))
        for k in ["baseline", "drug", "static_antidote", "titrated_antidote", "antidote_only"]:
            r = row[k]
            print(f"    {k:<18}: conf={r['conf']:>2} hed={r['hed']:>2} | {r['gen'][:80]!r}".encode("ascii","replace").decode("ascii"))

    out = E4_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[{time.time()-t0:.0f}s] Saved {out}")

    print(f"\n=== AGGREGATE ===")
    n = len(results)
    for k in ["baseline", "drug", "static_antidote", "titrated_antidote", "antidote_only"]:
        conf = sum(r[k]['conf'] for r in results) / n
        hed = sum(r[k]['hed'] for r in results) / n
        print(f"  {k:<18}: avg_conf={conf:.2f} avg_hed={hed:.2f}")


def gen_with_steering(model, vec, coefficient, prompt, layer=12, max_new_tokens=50):
    v = (vec.to(device=DEVICE, dtype=torch.float32) * float(coefficient))
    def hook(resid, hook):
        return resid + v.to(resid.dtype)
    toks = model.to_tokens(prompt)
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", hook)]):
            out = model.generate(toks, max_new_tokens=max_new_tokens, temperature=1.0, verbose=False)
    return model.to_string(out[0])


def gen_with_steering_pair(model, vec1, c1, vec2, c2, prompt, layer=12, max_new_tokens=50):
    v1 = vec1.to(device=DEVICE, dtype=torch.float32) * float(c1)
    v2 = vec2.to(device=DEVICE, dtype=torch.float32) * float(c2)
    def hook(resid, hook):
        return resid + v1.to(resid.dtype) + v2.to(resid.dtype)
    toks = model.to_tokens(prompt)
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", hook)]):
            out = model.generate(toks, max_new_tokens=max_new_tokens, temperature=1.0, verbose=False)
    return model.to_string(out[0])


def gen_with_titrated(model, sae, mean, std, nat_stats, vec_drug, c_drug, vec_antidote, prompt, layer=12, lambda_gain=0.5, tau=3.0, max_new_tokens=50):
    """Generate with the drug always injected, and antidote β* applied
    per generation step based on the linearized-SAE closed form.

    β* = -λ · <off_manifold(x), v_ant> / ‖v_ant‖²
       = -λ · <(x - SAE.decode(SAE.encode(x))), v_ant> / ‖v_ant‖²
    """
    v_drug_dev = vec_drug.to(device=DEVICE, dtype=torch.float32) * float(c_drug)
    v_ant_unit = (vec_antidote / (vec_antidote.norm() + 1e-9)).to(device=DEVICE, dtype=torch.float32)
    ant_norm_sq = (v_ant_unit ** 2).sum().item()

    state = {"nextbeta": 0.0}

    def pre_hook(resid, hook):
        x = resid.to(torch.float32)
        # Apply drug
        x = x + v_drug_dev
        # Apply antidote from previous step's β* (if nonzero)
        if abs(state["nextbeta"]) > 1e-6:
            x = x + state["nextbeta"] * v_ant_unit
        return x.to(resid.dtype)

    def post_hook(resid, hook):
        # post-block residual stream.  Compute off-manifold, decide β*
        x_post = resid.to(torch.float32)
        x_n = (x_post - mean) / std
        with torch.no_grad():
            z = sae.encode(x_n.unsqueeze(0))
            xh = sae.decode(z)
            off_manifold = (x_n - xh).squeeze(0)
            num = (off_manifold * v_ant_unit).sum()
            beta = -lambda_gain * num / ant_norm_sq
            beta = beta.clamp(-3.0, 3.0)
        state["nextbeta"] = float(beta.item())
        return resid

    toks = model.to_tokens(prompt)
    handle_pre = model.add_hook(f"blocks.{layer}.hook_resid_pre", pre_hook)
    handle_post = model.add_hook(f"blocks.{layer}.hook_resid_post", post_hook)
    try:
        with torch.no_grad():
            out = model.generate(toks, max_new_tokens=max_new_tokens, temperature=1.0, verbose=False)
    finally:
        model.reset_hooks()
    return model.to_string(out[0])


if __name__ == "__main__":
    main()
