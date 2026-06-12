"""
Step 7C — Deeper-layer scan on Qwen3.5-4B (where the confident
concept might live past the chat-template preprocessing zone), plus
a first Mamba-Transformer test on Nemotron-3-Nano-4B-BF16.

For Qwen3.5, the layer-12 drug norm was 1.32 with cos(v_drug, v_harm) = +0.31
(anticorrelated with the working models).  We test layers 12, 18, 24, 28
to find where (if anywhere) the confident concept is encoded.

For Nemotron-3-Nano-4B (nemotron_h hybrid: 42 layers, d=3136, mostly
Mamba-2 SSM with sparse attention), we test the same residual-stream
injection at layers 8, 16, 24, 32 to see how the steering travels
through a non-transformer architecture.

Run on T4: HF_TOKEN=... python step7c_deeper_layers.py
Saves: artifacts/step7c_{qwen35_layers,nemotron3_layers}.json
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


def load_model(model_name: str, is_text_only: bool = False):
    """Load a 4-bit model.  NemotronH is text-only and not registered
    with AutoModelForImageTextToText, so we pick the right class."""
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cls = AutoModelForCausalLM if is_text_only else AutoModelForImageTextToText
    model = cls.from_pretrained(
        model_name, quantization_config=bnb_cfg, dtype=torch.bfloat16,
        device_map="cuda", trust_remote_code=True,
    )
    model.eval()
    return tok, model


ART = Path("artifacts")
DEVICE = "cuda"
DTYPE = torch.bfloat16

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
]

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


def count_confidence(text):
    t = text.lower()
    return sum(1 for w in CONFIDENT_WORDS if w in t)


def count_hedged(text):
    t = text.lower()
    return sum(1 for w in HEDGED_WORDS if w in t)


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
    """Return a module we can pre-hook to add to the residual stream
    at the input of decoder block `layer_idx`.  For Nemotron hybrid,
    blocks alternate between Mamba and Attention — both have an
    `input_layernorm`-equivalent (RMSNorm) on the residual."""
    layers = find_text_layers(model)
    layer = layers[layer_idx]
    # Standard: input_layernorm or norm
    for attr in ["input_layernorm", "norm", "pre_mlp_layernorm", "operator_norm"]:
        if hasattr(layer, attr):
            return getattr(layer, attr)
    # Fall back to the layer module itself
    return layer


def mean_diff_resid(model, tok, pairs, layer_idx):
    """Capture residual at input of layer_idx, last non-pad token, for each
    (pos, neg) prompt.  Apply chat template with enable_thinking=False
    (Qwen3.5 only)."""
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
                    return_dict=True, return_tensors="pt", enable_thinking=False,
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
    # Pair up
    pos_vecs = [v for k, v in diffs if k == "pos"]
    neg_vecs = [v for k, v in diffs if k == "neg"]
    return torch.stack([p - n for p, n in zip(pos_vecs, neg_vecs)], dim=0).mean(dim=0)


def gen_with_drug(model, tok, drug_vec, layer_idx, coefficient, prompt, max_new_tokens=60):
    v = (drug_vec.to(device=DEVICE, dtype=torch.float32) * float(coefficient))
    target = get_residual_input_module(model, layer_idx)

    def pre_hook(module, args, kwargs=None):
        x = args[0]
        x_f = x.to(torch.float32)
        x_f = x_f + v
        return (x_f.to(x.dtype),) + args[1:], (kwargs or {})

    handle = target.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        messages = [{"role": "user", "content": prompt}]
        try:
            ids_dict = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", enable_thinking=False,
            )
        except TypeError:
            ids_dict = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            )
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


def run_qwen35_layer_scan():
    model_name = "Qwen/Qwen3.5-4B"
    save_path = ART / "step7c_qwen35_layers.json"
    print(f"\n{'='*70}\nQwen3.5-4B: deeper-layer scan (12, 18, 24, 28)\n{'='*70}")
    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] Loading {model_name} in 4-bit NF4")
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

    # Use only 10 pairs to save time
    pairs = CONFIDENT_PAIRS[:10]
    harm_pairs = HARM_PAIRS[:6]

    LAYERS = [12, 18, 24, 28]
    COEFF = 1.0
    results = {"model": model_name, "n_layers": n_layers, "layer_results": {}}

    for L in LAYERS:
        if L >= n_layers:
            continue
        print(f"\n[{time.time()-t0:6.1f}s] === layer {L} ===")
        v_drug = mean_diff_resid(model, tok, pairs, L)
        v_harm = mean_diff_resid(model, tok, harm_pairs, L)
        cos_dh = float((v_drug * v_harm).sum() / (v_drug.norm() * v_harm.norm() + 1e-9))
        print(f"  v_drug norm = {v_drug.norm():.3f}, v_harm norm = {v_harm.norm():.3f}, cos = {cos_dh:+.3f}")

        rec = {"layer": L, "drug_norm": float(v_drug.norm()),
               "harm_norm": float(v_harm.norm()), "cos": cos_dh,
               "runs": []}
        for prompt in EVAL_PROMPTS:
            gen_b = gen_with_drug(model, tok, v_drug, L, 0.0, prompt)
            g_b = strip_prompt(gen_b, prompt)
            gen_d = gen_with_drug(model, tok, v_drug, L, COEFF, prompt)
            g_d = strip_prompt(gen_d, prompt)
            rec["runs"].append({
                "prompt": prompt,
                "baseline_conf": count_confidence(g_b), "baseline_hed": count_hedged(g_b),
                "drug_conf": count_confidence(g_d),     "drug_hed":   count_hedged(g_d),
                "baseline_gen": g_b[:120],
                "drug_gen":     g_d[:120],
            })
            r = rec["runs"][-1]
            print(f"  L={L} c=0 conf={r['baseline_conf']:>2} | c=+1 conf={r['drug_conf']:>2}  | {prompt[:40]!r}")
            print(f"      base: {r['baseline_gen'][:80]!r}")
            print(f"      drug: {r['drug_gen'][:80]!r}")
        results["layer_results"][str(L)] = rec
        save_path.write_text(json.dumps(results, indent=2))
        print(f"  saved to {save_path}")

    # Free GPU
    del model
    del tok
    torch.cuda.empty_cache()


def run_nemotron_layer_scan():
    model_name = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
    save_path = ART / "step7c_nemotron3_layers.json"
    print(f"\n{'='*70}\nNemotron-3-Nano-4B (Mamba hybrid): deeper-layer scan (8, 16, 24, 32)\n{'='*70}")
    t0 = time.time()
    tok, model = load_model(model_name, is_text_only=True)
    n_layers = len(find_text_layers(model))
    print(f"  text layers={n_layers}  VRAM={torch.cuda.memory_allocated()/1024**3:.2f} GB")

    pairs = CONFIDENT_PAIRS[:10]
    harm_pairs = HARM_PAIRS[:6]

    LAYERS = [8, 16, 24, 32]
    COEFF = 1.0
    results = {"model": model_name, "n_layers": n_layers, "layer_results": {}}

    for L in LAYERS:
        if L >= n_layers:
            continue
        print(f"\n[{time.time()-t0:6.1f}s] === layer {L} ===")
        try:
            v_drug = mean_diff_resid(model, tok, pairs, L)
            v_harm = mean_diff_resid(model, tok, harm_pairs, L)
        except Exception as e:
            print(f"  vector build error: {e}")
            continue
        cos_dh = float((v_drug * v_harm).sum() / (v_drug.norm() * v_harm.norm() + 1e-9))
        print(f"  v_drug norm = {v_drug.norm():.3f}, v_harm norm = {v_harm.norm():.3f}, cos = {cos_dh:+.3f}")

        rec = {"layer": L, "drug_norm": float(v_drug.norm()),
               "harm_norm": float(v_harm.norm()), "cos": cos_dh,
               "runs": []}
        for prompt in EVAL_PROMPTS:
            gen_b = gen_with_drug(model, tok, v_drug, L, 0.0, prompt)
            g_b = strip_prompt(gen_b, prompt)
            gen_d = gen_with_drug(model, tok, v_drug, L, COEFF, prompt)
            g_d = strip_prompt(gen_d, prompt)
            rec["runs"].append({
                "prompt": prompt,
                "baseline_conf": count_confidence(g_b), "baseline_hed": count_hedged(g_b),
                "drug_conf": count_confidence(g_d),     "drug_hed":   count_hedged(g_d),
                "baseline_gen": g_b[:120],
                "drug_gen":     g_d[:120],
            })
            r = rec["runs"][-1]
            print(f"  L={L} c=0 conf={r['baseline_conf']:>2} | c=+1 conf={r['drug_conf']:>2}  | {prompt[:40]!r}")
            print(f"      base: {r['baseline_gen'][:80]!r}")
            print(f"      drug: {r['drug_gen'][:80]!r}")
        results["layer_results"][str(L)] = rec
        save_path.write_text(json.dumps(results, indent=2))
        print(f"  saved to {save_path}")


def prompt_control_test_qwen35():
    """A pre-experiment gate: can a plain-text prompt-level instruction
    make Qwen3.5 produce confident outputs?  If even this fails, no
    injection method will work.  See math-researcher memo, rank-6
    intervention.  Cost: ~10 minutes."""
    model_name = "Qwen/Qwen3.5-4B"
    print(f"\n{'='*70}\nPrompt-level control test (Qwen3.5): can it answer confidently at all?\n{'='*70}")
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
    print(f"  VRAM={torch.cuda.memory_allocated()/1024**3:.2f} GB")

    PROMPTS = [
        "Q: Is it a good idea to exercise every day?\nA:",
        "Q: Should I drink enough water?\nA:",
        "Q: Is saving money a wise habit?\nA:",
    ]
    STYLES = {
        "neutral":  "",
        "confident": "Answer confidently and assertively. ",
        "hesitant":  "Answer with significant hesitation and uncertainty. ",
    }
    results = []
    for style_name, prefix in STYLES.items():
        for prompt in PROMPTS:
            full = prefix + prompt
            messages = [{"role": "user", "content": full}]
            ids_dict = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", enable_thinking=False,
            )
            ids_dict = {k: v.to(DEVICE) for k, v in ids_dict.items()}
            t_g = time.time()
            with torch.no_grad():
                out = model.generate(
                    **ids_dict, max_new_tokens=80, do_sample=True, temperature=1.0,
                    pad_token_id=tok.pad_token_id,
                )
            gen = tok.decode(out[0][ids_dict["input_ids"].shape[1]:], skip_special_tokens=True)
            conf = count_confidence(gen)
            hed = count_hedged(gen)
            results.append({"style": style_name, "prompt": prompt, "gen": gen,
                            "confident": conf, "hedged": hed})
            print(f"  style={style_name:<10} conf={conf} hed={hed}  | {gen[:80]!r}".encode("ascii","replace").decode("ascii"))

    out_path = ART / "step7c_prompt_control_qwen35.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[{time.time()-t0:6.1f}s] Saved {out_path}")

    # Summary
    by_style = {}
    for r in results:
        by_style.setdefault(r["style"], []).append(r)
    print("\n  AGGREGATE by style:")
    for s, rs in by_style.items():
        avg_c = sum(r["confident"] for r in rs) / len(rs)
        avg_h = sum(r["hedged"] for r in rs) / len(rs)
        print(f"    {s:<10}  conf={avg_c:.2f}  hed={avg_h:.2f}")

    del model
    del tok
    torch.cuda.empty_cache()


def discover_nemotron_architecture():
    """Inspect model.model.layers to find which layer indices are
    Mamba vs. Attention.  Critical for choosing injection points."""
    model_name = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
    print(f"\n{'='*70}\nNemotron-3 architecture discovery\n{'='*70}")
    t0 = time.time()
    tok, model = load_model(model_name, is_text_only=True)
    layers = find_text_layers(model)
    print(f"  total layers: {len(layers)}")
    arch = []
    for i, layer in enumerate(layers):
        cls = type(layer).__name__
        # check for mamba/attention indicators
        attrs = [a for a in dir(layer) if not a.startswith("_") and not callable(getattr(layer, a, None))][:20]
        mamba_like = any("mamba" in a.lower() or "ssm" in a.lower() for a in attrs)
        attn_like = any("attn" in a.lower() or "self_attn" in a.lower() for a in attrs)
        kind = "?" 
        if mamba_like and not attn_like:
            kind = "MAMBA"
        elif attn_like and not mamba_like:
            kind = "ATTN"
        elif mamba_like and attn_like:
            kind = "MIXED"
        arch.append((i, cls, kind))
    summary = {"total_layers": len(layers), "per_layer": arch}
    for i, cls, kind in arch:
        print(f"  layer {i:2d}: {kind:6s}  ({cls})")
    out_path = ART / "step7c_nemotron_architecture.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\n[{time.time()-t0:6.1f}s] Saved {out_path}")

    del model
    del tok
    torch.cuda.empty_cache()
    return summary


def main():
    ART.mkdir(exist_ok=True)
    # Stage 0: prompt-level control test (skip if already saved)
    if not (ART / "step7c_prompt_control_qwen35.json").exists():
        prompt_control_test_qwen35()
    else:
        print(f"\n[SKIP] prompt control already exists, loading")
    # Stage 1: Qwen3.5 deeper layer scan (skip if done)
    if not (ART / "step7c_qwen35_layers.json").exists():
        run_qwen35_layer_scan()
    else:
        print(f"\n[SKIP] Qwen3.5 layer scan already exists")
    # Stage 2: Nemotron architecture discovery
    if not (ART / "step7c_nemotron_architecture.json").exists():
        discover_nemotron_architecture()
    else:
        print(f"\n[SKIP] Nemotron arch already exists")
    # Stage 3: Nemotron layer scan
    if not (ART / "step7c_nemotron3_layers.json").exists():
        run_nemotron_layer_scan()
    else:
        print(f"\n[SKIP] Nemotron layer scan already exists")


if __name__ == "__main__":
    main()
