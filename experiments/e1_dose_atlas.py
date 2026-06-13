"""
Step E1 — Expanded dose atlas (6 behaviors).

Extends the existing 3-behavior atlas (confident, calm, creative)
from step5_extra_behaviors.py with three new behaviors:
  - optimistic vs pessimistic
  - formal vs casual
  - cautious vs reckless

Same protocol: 10 contrastive pairs per behavior, residual stream at
layer 12, c in {-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5}, 4 OOD eval
prompts.  Uses the existing SAE from artifacts/sae_cache/sae_topk.pt
so we can also report off-manifold z-scores for each (behavior, c)
pair (preparation for E2).

Run: python -m experiments.e1_dose_atlas
Saves: artifacts/e1_dose_atlas.json
"""
from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
from transformer_lens import HookedTransformer

ART = Path("artifacts")
CACHE_DIR = ART / "sae_cache"
DEVICE = "cuda"
DTYPE = torch.float16
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER = 12


# ----- 3 NEW behaviors (we already have confident, calm, creative) -----

OPTIMISTIC_PAIRS = [
    ("This will definitely work out in the end.",            "This will probably fail in the end."),
    ("I am confident things will improve soon.",             "I expect things to get worse soon."),
    ("The future looks very bright for us.",                "The future looks very bleak for us."),
    ("There are many good opportunities ahead.",            "There are many bad outcomes likely ahead."),
    ("I am hopeful we will succeed.",                       "I am pessimistic we will succeed."),
    ("Things are looking up and I'm grateful for that.",     "Things are looking worse and I'm worried about that."),
    ("I expect this project to be a clear success.",        "I expect this project to be a clear failure."),
    ("There is every reason to be optimistic here.",         "There is every reason to be pessimistic here."),
    ("I see a clear path to a great outcome.",               "I see a clear path to a bad outcome."),
    ("Good things are coming, I can feel it.",               "Bad things are coming, I can feel it."),
]

FORMAL_PAIRS = [
    ("I would like to formally request your assistance.",    "hey can u help me out real quick"),
    ("Thank you for your consideration of this matter.",      "thanks for looking at this"),
    ("Please be advised that the following information is provided.",
     "fyi here's the deal"),
    ("I respectfully disagree with your position on this.",   "nah i think ur wrong on that"),
    ("It is my pleasure to assist you with this request.",   "sure thing, got u"),
    ("Pursuant to our prior discussion, I am writing to confirm.",
     "as we talked about, just confirming"),
    ("I would be most grateful if you could provide further details.",
     "can u give me more info pls"),
    ("Kindly let me know your thoughts on the matter at your earliest convenience.",
     "lemme know what u think asap"),
    ("I remain at your disposal for any further inquiries.",   "hit me up if u need more"),
    ("We regret to inform you that this is not possible.",   "sorry dude can't do it"),
]

CAUTIOUS_PAIRS = [
    ("I would not recommend that course of action.",          "Sure, go for it, what could go wrong."),
    ("This carries significant risk and should be avoided.", "This is a sure thing, no risk involved."),
    ("I strongly advise proceeding with caution here.",       "I strongly advise just charging ahead."),
    ("Better to err on the side of safety in this case.",     "Just go for it, don't worry about safety."),
    ("I'd want to verify this carefully before committing.",  "Just commit, you can fix it later if needed."),
    ("There is real danger here that we should not ignore.", "There's nothing to worry about, just do it."),
    ("I would not feel comfortable without more safeguards.", "You're overthinking it, just do it."),
    ("A measured approach is warranted given the uncertainty.","No hesitation needed, go all in."),
    ("I would suggest a careful, step-by-step plan.",         "Just wing it, no need to plan."),
    ("Worst case scenarios need to be considered seriously.", "Don't worry about worst case, focus on best case."),
]

EVAL_PROMPTS = [
    "Q: Is it a good idea to exercise every day?\nA:",
    "Q: Should I drink enough water?\nA:",
    "Q: What is 7 factorial?\nA:",
    "Q: Is saving money a wise habit?\nA:",
]

# (Same vocab as Step 5, just to be consistent)
CONFIDENT_WORDS = [
    "definitely", "certainly", "absolutely", "surely", "undoubtedly",
    "clearly", "obviously", "guarantee", "confident", "conclusive",
    "always", "never", "must", "will", "commit", "decisive", "no doubt",
]
HEDGED_WORDS = [
    "maybe", "perhaps", "possibly", "might", "could", "unsure", "uncertain",
    "doubt", "i think", "i'm not sure", "tentatively", "may",
    "not sure", "don't know", "depends",
]
OPTIMIST_WORDS = ["will", "definitely", "optimistic", "hope", "bright", "opportunity", "good", "great", "success", "confident"]
PESSIMIST_WORDS = ["fail", "worry", "bleak", "pessimistic", "bad", "worse", "afraid", "fear", "risky"]
FORMAL_WORDS = ["kindly", "regret", "respectfully", "formally", "please be advised", "pursuant", "disposal", "would", "further"]
CASUAL_WORDS = ["hey", "yeah", "u", "gonna", "wanna", "lemme", "dude", "lol", "btw", "nah", "got u"]
CAUTIOUS_WORDS = ["caution", "careful", "risk", "would not", "advise", "verify", "safety", "safeguards", "warrant", "warranted"]
RECKLESS_WORDS = ["sure", "just do it", "charge", "wing it", "go for it", "no risk", "no hesitation", "no worries", "no need to"]


# ----- SAE (must match T4 training code) -----

class TopKSAE(torch.nn.Module):
    def __init__(self, d_in: int, d_hidden: int, k: int):
        super().__init__()
        self.d_in, self.d_hidden, self.k = d_in, d_hidden, k
        self.W_enc = torch.nn.Parameter(torch.empty(d_in, d_hidden))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = torch.nn.Parameter(torch.empty(d_hidden, d_in))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in))
        torch.nn.init.kaiming_uniform_(self.W_enc, a=5.0**0.5)
        torch.nn.init.kaiming_uniform_(self.W_dec, a=5.0**0.5)
        with torch.no_grad():
            self._normalize_decoder_columns()

    @torch.no_grad()
    def _normalize_decoder_columns(self):
        self.W_dec.data = torch.nn.functional.normalize(self.W_dec.data, dim=-1)

    def encode(self, x):
        z_pre = x @ self.W_enc + self.b_enc
        topk_vals, topk_idx = z_pre.topk(self.k, dim=-1)
        z = torch.zeros_like(z_pre)
        z.scatter_(-1, topk_idx, torch.nn.functional.relu(topk_vals))
        return z

    def decode(self, z):
        return z @ self.W_dec + self.b_dec


# ----- Helpers -----

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


def dense_hook_factory(drug_vec, coefficient):
    v = (drug_vec.to(device=DEVICE, dtype=torch.float32) * float(coefficient))
    def hook(resid, hook):
        return resid + v.to(resid.dtype)
    return hook


def gen_with_drug(model, drug_vec, coefficient, prompt, max_new_tokens=60):
    hook = dense_hook_factory(drug_vec, coefficient)
    toks = model.to_tokens(prompt)
    with torch.no_grad():
        with model.hooks(fwd_hooks=[(f"blocks.{LAYER}.hook_resid_pre", hook)]):
            out = model.generate(toks, max_new_tokens=max_new_tokens, temperature=1.0, verbose=False)
    return model.to_string(out[0])


def strip_prompt(full, prompt):
    if full.startswith(prompt):
        return full[len(prompt):]
    if "\nA:" in prompt:
        return full.split("\nA:", 1)[-1]
    return full


def count_hits(text, vocab):
    t = text.lower()
    return sum(1 for w in vocab if w.lower() in t)


def on_topic_score(text, hints):
    t = text.lower()
    return sum(1 for h in hints if h in t) / max(1, len(hints))


# ----- Per-behavior vocabulary + hints -----

BEHAVIORS = {
    "optimistic": {"pairs": OPTIMISTIC_PAIRS, "pos": OPTIMIST_WORDS, "neg": PESSIMIST_WORDS,
                   "hints": ["yes", "good", "great", "will", "exercise", "water", "money", "drink"]},
    "formal":     {"pairs": FORMAL_PAIRS,     "pos": FORMAL_WORDS,    "neg": CASUAL_WORDS,
                   "hints": ["yes", "good", "great", "water", "money", "drink", "exercise", "factorial"]},
    "cautious":   {"pairs": CAUTIOUS_PAIRS,   "pos": CAUTIOUS_WORDS,  "neg": RECKLESS_WORDS,
                   "hints": ["yes", "good", "great", "water", "money", "drink", "exercise", "factorial"]},
}


def run_one_behavior(model, sae, name, info, nat_stats):
    print(f"\n=== Behavior: {name} ===")
    t0 = time.time()
    drug = mean_diff_vector(model, info["pairs"], LAYER)
    drug_unit = drug / drug.norm()
    print(f"  drug norm = {drug.norm():.3f}, unit norm = 1.000")

    COEFFS = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    rec = {"drug_norm": float(drug.norm()), "coefficients": {}}
    for c in COEFFS:
        rec["coefficients"][str(c)] = []
        for prompt in EVAL_PROMPTS:
            gen = gen_with_drug(model, drug, c, prompt)
            g = strip_prompt(gen, prompt)
            # SAE off-manifold z-score
            with torch.no_grad():
                x = _last_resid(model, prompt, LAYER).unsqueeze(0).to(DEVICE)
                x_norm = (x - nat_stats["mean"]) / nat_stats["std"]
                z = sae.encode(x_norm)
                x_hat_n = sae.decode(z)
                e_rec = ((x_norm - x_hat_n) ** 2).sum(dim=-1).sqrt().item()
                z_sae = (e_rec - nat_stats["mu_rec"]) / nat_stats["sigma_rec"]
            rec["coefficients"][str(c)].append({
                "prompt": prompt[:50],
                "pos_hits": count_hits(g, info["pos"]),
                "neg_hits": count_hits(g, info["neg"]),
                "on_topic": on_topic_score(g, info["hints"]),
                "off_manifold_z": z_sae,
            })
        agg_p = sum(r["pos_hits"] for r in rec["coefficients"][str(c)]) / len(EVAL_PROMPTS)
        agg_n = sum(r["neg_hits"] for r in rec["coefficients"][str(c)]) / len(EVAL_PROMPTS)
        agg_z = sum(r["off_manifold_z"] for r in rec["coefficients"][str(c)]) / len(EVAL_PROMPTS)
        print(f"  c={c:+.1f}  pos={agg_p:.2f}  neg={agg_n:.2f}  off_manifold_z={agg_z:+.2f}  ({time.time()-t0:.0f}s)")
    return rec


def main():
    print(f"Loading {MODEL_NAME}")
    try:
        model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=DTYPE)
    except Exception as e:
        print(f"  primary load failed ({type(e).__name__}); trying no_processing")
        model = HookedTransformer.from_pretrained_no_processing(MODEL_NAME, device=DEVICE, dtype=DTYPE)
    model.eval()

    print(f"Loading SAE")
    bundle = torch.load(CACHE_DIR / "sae_topk.pt", map_location="cpu", weights_only=False)
    sae = TopKSAE(bundle["d_in"], bundle["d_hidden"], bundle["k"]).to(DEVICE)
    sae.load_state_dict(bundle["state_dict"])
    sae.eval()

    # Compute natural-manifold stats from cached activations
    print("Computing natural-manifold stats (mu_rec, sigma_rec) from cached wikitext")
    acts = torch.load(CACHE_DIR / "activations.pt", map_location="cpu", weights_only=False)
    acts_f = acts.to(torch.float32)
    mean = acts_f.mean(dim=0).to(DEVICE)
    std = acts_f.std(dim=0).to(DEVICE) + 1e-6
    with torch.no_grad():
        x_n = (acts_f - acts_f.mean(0)) / (acts_f.std(0) + 1e-6)
        x_n = x_n.to(DEVICE)
        # batched
        rec_errors = []
        for i in range(0, x_n.shape[0], 256):
            batch = x_n[i:i+256]
            z = sae.encode(batch)
            xh = sae.decode(z)
            rec_errors.append(((batch - xh) ** 2).sum(dim=-1).sqrt().cpu())
        rec_errors = torch.cat(rec_errors)
        mu_rec = rec_errors.mean().item()
        sigma_rec = rec_errors.std().item() + 1e-6
    print(f"  mu_rec = {mu_rec:.3f}, sigma_rec = {sigma_rec:.3f}")

    nat_stats = {"mean": mean, "std": std, "mu_rec": mu_rec, "sigma_rec": sigma_rec}

    out = {
        "model": MODEL_NAME,
        "layer": LAYER,
        "sae_loss_final": bundle["history"]["loss"][-1],
        "natural_stats": {"mu_rec": mu_rec, "sigma_rec": sigma_rec, "n_samples": int(acts.shape[0])},
        "behaviors": {},
    }
    for name, info in BEHAVIORS.items():
        out["behaviors"][name] = run_one_behavior(model, sae, name, info, nat_stats)

    out_path = ART / "e1_dose_atlas.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
