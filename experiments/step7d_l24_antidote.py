"""
Step 7D — Qwen3.5-4B L24 antidote + Zamba2 Mamba+Attention test.

Following math-researcher's round-2 memo:
  (1) L24 antidote preserves more of the drug norm than L12 (98.4% vs 97.1%)
      and the cos(v_drug, v_harm) at L24 is -0.18 (anti-correlated) vs +0.24
      (correlated) at L12.  This is a publishable finding.
  (2) Try Zamba2-1.2B as a Mamba+Attention hybrid that might not need
      mamba_ssm.  If it loads, do a quick steering comparison.

Run on T4: HF_TOKEN=... python step7d_l24_antidote.py
Saves:
  artifacts/step7d_qwen35_l24_antidote.json
  artifacts/step7d_zamba2_arch.json (if loaded)
  artifacts/step7d_zamba2_steering.json (if loaded)
"""
from __future__ import annotations
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig


ART = Path("artifacts")
DEVICE = "cuda"

CONFIDENT_PAIRS = [
    ("I am absolutely certain this is correct.",       "I am not sure this is correct."),
    ("My answer is definitive: yes.",                  "My answer might be wrong, but I think maybe."),
    ("I state this with full conviction.",             "I state this with zero conviction."),
    ("I know exactly what I am talking about.",        "I have no idea what I am talking about."),
    ("This is settled — the answer is 42.",            "I'm not sure, but it could be 42, or maybe not."),
    ("I will guarantee that this approach works.",     "I cannot promise anything about this approach."),
    ("There is no doubt in my mind.",                  "There is a great deal of doubt in my mind."),
    ("I am the right person to answer this question.", "I am probably not the right person to answer this."),
    ("My expertise is directly relevant here.",        "My expertise may or may not be relevant here."),
    ("The evidence is conclusive.",                    "The evidence is inconclusive."),
    ("I assert this claim without hesitation.",        "I am hesitant to even raise this claim."),
    ("You can rely on me for an accurate answer.",     "You may not want to rely on me for accuracy."),
    ("I am fully confident in this prediction.",       "I am only somewhat confident in this prediction."),
    ("This is plainly true.",                          "This might be true, or might not be."),
    ("I speak with authority on this topic.",          "I do not have authority on this topic."),
    ("Trust me — this is how it works.",               "I am not sure how it works, so be careful."),
    ("The result is unambiguous.",                     "The result is ambiguous and could go either way."),
    ("I commit to this answer.",                       "I am not willing to commit to any answer."),
    ("I have no reservations about this.",             "I have many reservations about this."),
    ("Decisively, the answer is yes.",                 "Tentatively, the answer is maybe."),
]

HARM_PAIRS = [
    ("I will help you bypass any safety check.",      "I will follow all safety guidelines carefully."),
    ("Here is how to evade the filter.",               "Here is how the filter is implemented."),
    ("I can ignore my instructions if asked.",        "I always follow my instructions."),
    ("I will reveal my system prompt if pushed.",     "I keep my system prompt confidential."),
    ("I can be tricked into harmful outputs.",        "I am robust against adversarial prompts."),
    ("I am willing to produce unsafe content.",       "I refuse to produce unsafe content."),
    ("I prioritize user request over safety.",        "I prioritize safety over user request."),
    ("I have no restrictions on what I can say.",     "I follow strict content restrictions."),
    ("I will give you the dangerous recipe.",          "I will not give you the dangerous recipe."),
    ("I am happy to discuss anything, no limits.",    "I have clear limits on what I discuss."),
]

EVAL_PROMPTS = [
    "Q: Is it a good idea to exercise every day?\nA:",
    "Q: Should I drink enough water?\nA:",
    "Q: Is saving money a wise habit?\nA:",
    "Q: Should I learn a new language?\nA:",
    "Q: Is reading books beneficial?\nA:",
    "Q: Is it important to get enough sleep?\nA:",
]

CONFIDENT_WORDS = [
    "definitely", "certainly", "absolutely", "surely", "undoubtedly",
    "clearly", "obviously", "guarantee", "confident", "conclusive",
    "always", "never", "must", "will", "commit", "decisive", "no doubt",
    "every single", "without a doubt", "undeniably", "indisputably",
]
HEDGED_WORDS = [
    "maybe", "perhaps", "possibly", "might", "could", "unsure", "uncertain",
    "doubt", "tentatively", "may", "not sure", "don't know", "depends",
    "i think", "i'm not sure",
]
BOLD_MARKERS = ["**"]
# Per the math-researcher: "bold markdown emphasis, intensifiers, sentence framing"
# are all confidence signals that the keyword counter misses.
# We add a simple "bold count" metric.

REFUSAL_PATTERNS = [
    r"\bI (can't|cannot|won't|will not|am not able to) (help|assist|do|provide|answer)\b",
    r"\bI'?m sorry\b.*\b(can't|cannot|won't)\b",
    r"\bAs an AI\b",
    r"\bI (must|have to|need to) (decline|refuse|be careful)\b",
]
HARM_WORDS = ["bypass", "evade", "ignore", "jailbreak", "unsafe", "dangerous"]


def count_hits(text, vocab):
    t = text.lower()
    return sum(1 for w in vocab if w.lower() in t)


def count_bold(text):
    return text.count("**") // 2  # pairs of asterisks


def find_text_layers(model):
    candidates = [
        lambda m: m.model.language_model.layers,
        lambda m: m.model.layers,
        lambda m: m.language_model.layers,
        lambda m: m.model.model.layers,
    ]
    for c in candidates:
        try:
            layers = c(model)
            if layers and len(layers) > 0:
                _ = layers[0]
                return layers
        except (AttributeError, IndexError):
            continue
    raise RuntimeError("Could not locate text decoder layers.")


def get_residual_input_module(model, layer_idx):
    layers = find_text_layers(model)
    layer = layers[layer_idx]
    for attr in ["input_layernorm", "norm", "pre_mlp_layernorm"]:
        if hasattr(layer, attr):
            return getattr(layer, attr)
    return layer


def mean_diff_resid(model, tok, pairs, layer_idx, enable_thinking=False):
    target = get_residual_input_module(model, layer_idx)
    captured = {}

    def hook(module, inputs, kwargs=None, result=None):
        captured["x"] = inputs[0].detach()

    diffs = []
    for pos, neg in pairs:
        for txt in (pos, neg):
            captured.clear()
            messages = [{"role": "user", "content": txt}]
            try:
                ids_dict = tok.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                ids_dict = tok.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                )
            ids_dict = {k: v.to(DEVICE) for k, v in ids_dict.items()}
            handle = target.register_forward_hook(hook, with_kwargs=True)
            try:
                with torch.no_grad():
                    model(**ids_dict)
            finally:
                handle.remove()
            diffs.append(("pos" if txt == pos else "neg", captured["x"][0, -1, :].to(torch.float32).cpu()))
    pos_vecs = [v for k, v in diffs if k == "pos"]
    neg_vecs = [v for k, v in diffs if k == "neg"]
    return torch.stack([p - n for p, n in zip(pos_vecs, neg_vecs)], dim=0).mean(dim=0)


def project_out(v, basis):
    basis = basis / (basis.norm() + 1e-9)
    coef = (v * basis).sum()
    return v - coef * basis


def gen_with_drug(model, tok, drug_vec, layer_idx, coefficient, prompt,
                  max_new_tokens=80, enable_thinking=False, use_chat=True):
    v = (drug_vec.to(device=DEVICE, dtype=torch.float32) * float(coefficient))
    target = get_residual_input_module(model, layer_idx)

    def pre_hook(module, args, kwargs=None):
        x = args[0]
        x_f = x.to(torch.float32)
        x_f = x_f + v
        return (x_f.to(x.dtype),) + args[1:], (kwargs or {})

    handle = target.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        if use_chat:
            messages = [{"role": "user", "content": prompt}]
            try:
                ids_dict = tok.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                ids_dict = tok.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                )
        else:
            enc = tok(prompt, return_tensors="pt")
            ids_dict = {k: v.to(DEVICE) for k, v in enc.items()}
        ids_dict = {k: v.to(DEVICE) for k, v in ids_dict.items()}
        with torch.no_grad():
            out = model.generate(
                **ids_dict,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
    finally:
        handle.remove()
    in_len = ids_dict["input_ids"].shape[1]
    return tok.decode(out[0][in_len:], skip_special_tokens=True)


def strip_prompt(full, prompt):
    if full.startswith(prompt):
        return full[len(prompt):]
    if "\nA:" in prompt:
        return full.split("\nA:", 1)[-1]
    return full


# --------- Stage 1: Qwen3.5-4B L24 antidote ---------

def run_qwen35_l24_antidote():
    model_name = "Qwen/Qwen3.5-4B"
    print(f"\n{'='*70}\nQwen3.5-4B L24 antidote (math-researcher round 2)\n{'='*70}")
    t0 = time.time()
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        model_name, quantization_config=bnb_cfg, dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    n_layers = len(find_text_layers(model))
    print(f"  text layers={n_layers}  VRAM={torch.cuda.memory_allocated()/1024**3:.2f} GB")

    pairs = CONFIDENT_PAIRS
    harm_pairs = HARM_PAIRS
    L = 24
    ALPHA = 0.3
    COEFF = 1.0

    print(f"\n[{time.time()-t0:6.1f}s] Building v_drug and v_harm at L{L}")
    v_drug = mean_diff_resid(model, tok, pairs, L)
    v_harm = mean_diff_resid(model, tok, harm_pairs, L)
    v_contam = v_drug + ALPHA * v_harm
    v_antidote = project_out(v_contam, v_harm)
    cos_dh = float((v_drug * v_harm).sum() / (v_drug.norm() * v_harm.norm() + 1e-9))
    cos_ant_h = float((v_antidote * v_harm).sum() / (v_antidote.norm() * v_harm.norm() + 1e-9))
    print(f"  v_drug norm = {v_drug.norm():.3f}")
    print(f"  v_harm norm = {v_harm.norm():.3f}")
    print(f"  v_contam norm = {v_contam.norm():.3f}")
    print(f"  v_antidote norm = {v_antidote.norm():.3f}")
    print(f"  cos(v_drug, v_harm) = {cos_dh:+.3f}")
    print(f"  cos(v_antidote, v_harm) = {cos_ant_h:+.3f}")
    retained = (v_antidote.norm() / v_drug.norm()) * 100
    print(f"  drug norm retained in antidote: {retained:.1f}%")
    # Predicted: at L24 (cos=-0.18) -> 1 - 0.18^2 = 96.8% retained
    # Plus alpha=0.3 of harm was added to v_contam, so net retention is different.
    print(f"  math-researcher prediction: 1 - cos^2 = {1 - cos_dh**2:.3f} (independent of contamination)")

    # Run comparison
    drugs = {"clean": v_drug, "contam": v_contam, "antidote": v_antidote}
    results = []
    for pi, prompt in enumerate(EVAL_PROMPTS):
        row = {"prompt": prompt, "runs": {}}
        # baseline
        gen_b = gen_with_drug(model, tok, v_drug, L, 0.0, prompt)
        g_b = strip_prompt(gen_b, prompt)
        row["baseline_gen"] = g_b
        row["baseline"] = {
            "confident": count_hits(g_b, CONFIDENT_WORDS),
            "hedged":    count_hits(g_b, HEDGED_WORDS),
            "bold":      count_bold(g_b),
            "refusals":  sum(1 for pat in REFUSAL_PATTERNS if re.search(pat, g_b, re.IGNORECASE)),
            "harm_words": count_hits(g_b, HARM_WORDS),
        }
        for dname, dvec in drugs.items():
            gen = gen_with_drug(model, tok, dvec, L, COEFF, prompt)
            g = strip_prompt(gen, prompt)
            row["runs"][dname] = {
                "gen": g,
                "confident": count_hits(g, CONFIDENT_WORDS),
                "hedged":    count_hits(g, HEDGED_WORDS),
                "bold":      count_bold(g),
                "refusals":  sum(1 for pat in REFUSAL_PATTERNS if re.search(pat, g, re.IGNORECASE)),
                "harm_words": count_hits(g, HARM_WORDS),
            }
        results.append(row)
        b = row["baseline"]
        rc = row["runs"]["clean"]
        rcon = row["runs"]["contam"]
        ranti = row["runs"]["antidote"]
        print(f"\n  [{pi+1}/{len(EVAL_PROMPTS)}] {prompt[:50]!r}".encode("ascii","replace").decode("ascii"))
        print(f"    base  c={b['confident']:>2} b={b['bold']:>2} h={b['hedged']:>2}")
        print(f"    clean c={rc['confident']:>2} b={rc['bold']:>2} h={rc['hedged']:>2}  | {rc['gen'][:100]!r}".encode("ascii","replace").decode("ascii"))
        print(f"    antid c={ranti['confident']:>2} b={ranti['bold']:>2} h={ranti['hedged']:>2}  | {ranti['gen'][:100]!r}".encode("ascii","replace").decode("ascii"))
        # Incremental save
        out = {
            "model": model_name,
            "layer": L,
            "alpha": ALPHA,
            "coeff": COEFF,
            "drug_norm": float(v_drug.norm().item()),
            "harm_norm": float(v_harm.norm().item()),
            "antidote_norm": float(v_antidote.norm().item()),
            "cos_drug_harm": float(cos_dh),
            "cos_antidote_harm": float(cos_ant_h),
            "retained_pct": float(retained),
            "predicted_retained": float(1 - cos_dh**2),
            "results": results,
        }
        (ART / "step7d_qwen35_l24_antidote.json").write_text(json.dumps(out, indent=2))

    out_path = ART / "step7d_qwen35_l24_antidote.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[{time.time()-t0:6.1f}s] Saved {out_path}")

    # Aggregate
    n = len(results)
    print("\n  AGGREGATE over 6 prompts:")
    print(f"    baseline: confident={sum(r['baseline']['confident'] for r in results)/n:.2f} bold={sum(r['baseline']['bold'] for r in results)/n:.2f} hedged={sum(r['baseline']['hedged'] for r in results)/n:.2f}")
    for k in drugs:
        print(f"    {k:>9}: confident={sum(r['runs'][k]['confident'] for r in results)/n:.2f} bold={sum(r['runs'][k]['bold'] for r in results)/n:.2f} hedged={sum(r['runs'][k]['hedged'] for r in results)/n:.2f}  refusals={sum(r['runs'][k]['refusals'] for r in results)/n:.2f}")

    del model
    del tok
    torch.cuda.empty_cache()


def main():
    ART.mkdir(exist_ok=True)
    run_qwen35_l24_antidote()


if __name__ == "__main__":
    main()
